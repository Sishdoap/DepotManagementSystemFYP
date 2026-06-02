"""Live operator dashboard for the depot traffic management system.

Reads from the SQLite database the simulator writes to. Displays:
  - waiting queue
  - per-gate status with current truck and live "camera feed"
  - recent container reads (raw vs recovered codes; recovery flagged)
  - per-gate throughput chart

Run with:
    streamlit run scripts/dashboard.py
"""

import sys
import time
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from sqlalchemy import select, text
from src.dashboard_io import read_gate_frame
from src.db import containers, events, gate_throughput, get_sim_clock, make_engine


DB_URL = "sqlite:///depot_live.db"
GATE_IDS = ["A", "B", "C"]
REFRESH_INTERVAL_S = 1.0   # how often Streamlit re-runs the script


# --- Streamlit page setup ---

st.set_page_config(
    page_title="Depot Live",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
    .small-font { font-size: 0.85em; color: #888; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Data access ---

@st.cache_resource
def get_engine():
    """One engine per Streamlit session, reused across reruns."""
    return make_engine(DB_URL)


def fetch_waiting(engine, sim_now: float) -> pd.DataFrame:
    sql = (
        "SELECT id, plate, arrived_at, "
        f"       ({sim_now} - arrived_at) AS waited_s "
        "FROM events WHERE status='waiting' ORDER BY arrived_at"
    )
    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def fetch_at_gate(engine) -> pd.DataFrame:
    sql = (
        "SELECT plate, assigned_gate, assigned_at, routing_reason "
        "FROM events WHERE status='at_gate' ORDER BY assigned_at"
    )
    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def fetch_recent_containers(engine, limit=20) -> pd.DataFrame:
    sql = f"""
        SELECT c.recorded_at, e.plate, e.assigned_gate AS gate,
               c.raw_code, c.recovered_code, c.is_valid, c.recovery_edits
        FROM containers c
        JOIN events e ON c.event_id = e.id
        ORDER BY c.recorded_at DESC
        LIMIT {limit}
    """
    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def fetch_throughput_summary(engine, sim_now: float, window_seconds=600) -> pd.DataFrame:
    cutoff = sim_now - window_seconds
    sql = f"""
        SELECT gate_id, SUM(trucks_served) AS trucks, AVG(mean_service_time) AS avg_service
        FROM gate_throughput
        WHERE window_start >= {cutoff}
        GROUP BY gate_id
        ORDER BY gate_id
    """
    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn)
    return df


def fetch_throughput_history(engine, sim_now: float, minutes=30) -> pd.DataFrame:
    cutoff = sim_now - minutes * 60
    sql = f"""
        SELECT window_start, gate_id, trucks_served
        FROM gate_throughput
        WHERE window_start >= {cutoff}
        ORDER BY window_start
    """
    with engine.begin() as conn:
        df = pd.read_sql_query(text(sql), conn)
    if not df.empty:
        # window_start is sim seconds since sim start, not unix time.
        # Convert to a relative "minutes ago" axis for display.
        df["minutes_ago"] = ((sim_now - df["window_start"]) / 60.0).round(0).astype(int)
    return df

def fetch_avg_wait_recent(engine, sim_now: float, window_seconds: int = 600) -> float | None:
    """Average wait time across trucks that completed in the last window.

    Wait time = assigned_at - arrived_at. We measure on completed events
    only because those are the ones with finalized wait values; trucks still
    in the queue have wait times that grow every second.
    """
    cutoff = sim_now - window_seconds
    sql = f"""
        SELECT AVG(assigned_at - arrived_at) AS avg_wait
        FROM events
        WHERE departed_at IS NOT NULL
          AND departed_at >= {cutoff}
    """
    with engine.begin() as conn:
        row = conn.execute(text(sql)).first()
    return float(row.avg_wait) if row and row.avg_wait is not None else None


# --- UI ---

st.title("🚛 Depot Traffic Management — Live")

engine = get_engine()

# Try to read data; gracefully handle missing DB.
try:
    sim_now = get_sim_clock(engine) or 0.0
    waiting = fetch_waiting(engine, sim_now)
    at_gate = fetch_at_gate(engine)
    recent = fetch_recent_containers(engine)
    tp_summary = fetch_throughput_summary(engine, sim_now)
    tp_history = fetch_throughput_history(engine, sim_now)
    avg_wait = fetch_avg_wait_recent(engine, sim_now)
except Exception as e:
    st.error(
        f"Cannot read database `{DB_URL}`. Is the simulator running?\n\n"
        f"Start it with: `python scripts/run_realtime_sim.py`\n\nError: {e}"
    )
    st.stop()


# --- Top metrics row ---

col1, col2, col3, col4 = st.columns(4)
col1.metric("Waiting trucks", len(waiting))
col2.metric("At gate", len(at_gate))
col3.metric(
    "Avg wait (10m)",
    f"{avg_wait:.0f}s" if avg_wait is not None else "—",
)
col4.metric(
    "Throughput (10 min)",
    int(tp_summary["trucks"].sum()) if not tp_summary.empty else 0,
)


# --- Gate strip with live camera feeds ---

st.subheader("Gates")
gate_cols = st.columns(3)
for col, gate_id in zip(gate_cols, GATE_IDS):
    with col:
        st.markdown(f"**Gate {gate_id}**")
        # Camera frame. Wrapped in try/except because even with atomic writes,
        # extremely fast refresh + slow disk could in principle produce a
        # partial read, and PIL is unforgiving about truncated PNGs.
        frame_bytes = read_gate_frame(gate_id)
        if frame_bytes:
            try:
                st.image(frame_bytes, width="stretch")
            except (OSError, Exception):
                st.markdown(
                    "<div style='background:#222; height:120px; display:flex; "
                    "align-items:center; justify-content:center; color:#888; "
                    "border-radius:6px;'>— refreshing... —</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='background:#222; height:120px; display:flex; "
                "align-items:center; justify-content:center; color:#666; "
                "border-radius:6px;'>— no frame yet —</div>",
                unsafe_allow_html=True,
            )
        # Current occupant.
        current = at_gate[at_gate["assigned_gate"] == gate_id]
        if not current.empty:
            row = current.iloc[0]
            st.markdown(f"🚛 **{row['plate']}**")
            st.caption(row["routing_reason"] or "—")
        else:
            st.markdown("_idle_")

        # Per-gate stats.
        stats = tp_summary[tp_summary["gate_id"] == gate_id]
        if not stats.empty:
            r = stats.iloc[0]
            avg_s = r["avg_service"] if r["avg_service"] is not None else 0
            st.caption(f"served (10m): **{int(r['trucks'])}** · avg service: {avg_s:.1f}s")


# --- Two-column layout: waiting queue + recent reads ---

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Waiting queue")
    if waiting.empty:
        st.caption("Queue is empty.")
    else:
        display = waiting[["plate", "waited_s"]].rename(
            columns={"plate": "Plate", "waited_s": "Waited (s)"}
        )
        st.dataframe(display, hide_index=True, width="stretch")

with col_right:
    st.subheader("Recent container reads")
    if recent.empty:
        st.caption("No reads yet.")
    else:
        display = recent.copy()

        def _format_final(row):
            code = row["recovered_code"]
            # pandas NaN is truthy; treat it the same as None.
            if pd.isna(code) or code is None or code == "":
                code_str = "(unrecoverable)"
            else:
                code_str = str(code)
            return f"✅ {code_str}" if row["is_valid"] == 1 else f"❌ {code_str}"

        display["recovered"] = display.apply(_format_final, axis=1)

        display["edits"] = display["recovery_edits"].apply(
            lambda x: "" if pd.isna(x) or x == 0 else f"{int(x)} edit(s)"
        )
        display = display[["plate", "gate", "raw_code", "recovered", "edits"]]
        display.columns = ["Plate", "Gate", "Raw OCR", "Final", "Recovery"]
        st.dataframe(display, hide_index=True, width="stretch")


# --- Throughput chart ---

st.subheader("Per-gate throughput (last 30 min)")
if tp_history.empty:
    st.caption("Not enough history yet.")
else:
    # Pivot for stacked chart.
    pivot = (
        tp_history.pivot_table(
            index="minutes_ago", columns="gate_id", values="trucks_served", aggfunc="sum"
        )
        .fillna(0)
        .sort_index(ascending=False)
    )
    st.bar_chart(pivot)


# --- Auto-refresh ---

st.caption(
    f"Auto-refreshing every {REFRESH_INTERVAL_S}s · "
    f"reading {DB_URL} · frames in `./frames/`"
)
time.sleep(REFRESH_INTERVAL_S)
st.rerun()