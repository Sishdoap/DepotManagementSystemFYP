"""Depot cameras dashboard: trucks identified by the live pipeline (FR8).

Shows what scripts/run_live.py writes: the cameras with their tracked
trucks, trucks in view now (provisional passes), and every finished truck
as a profile card — container code(s) with how they were established,
size/type, vehicle type, capacity class, plates, and a picture per camera.

It works the same whether the pipeline follows the live streams or
replays a recorded hour (`run_live.py --replay <hour> --speed 1`, the
demo fallback); the status line says which.

Plates are personal data: they are masked unless "Show plates" is ticked
(keep it off for screenshots in the report).

Run alongside the pipeline:
    python scripts/run_live.py                       # or --replay <hour folder> --speed 1
    streamlit run scripts/depot_dashboard.py --server.address 127.0.0.1   # this machine only

(The gate simulation dashboard is scripts/dashboard.py.)
"""

import sys
import time
from datetime import datetime
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from src.dashboard_io import mask_plate, read_camera_preview, read_status, thumbnail_path
from src.db import init_schema, list_passes, make_engine

CAMERAS = [("561", "cab, barrier"), ("557", "side, entering"), ("560", "side, leaving"),
           ("558", "rear, top"), ("559", "rear, backup")]
STATUS_MARK = {"verified": "✅ verified", "recovered": "🛠️ recovered", "inferred": "❔ inferred (unverified)",
               "failed": "—"}
CAPACITY_MARK = {"light": "🟢 light", "medium": "🟡 medium", "heavy": "🔴 heavy", "unknown": "⚪ unknown"}

st.set_page_config(page_title="Depot cameras", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.header("Settings")
    db_url = st.text_input("Database", "sqlite:///depot_passes.db")
    live_dir = Path(st.text_input("Live files", "frames/live"))
    show_plates = st.checkbox("Show plates", value=False,
                              help="Plates are personal data; keep masked for screenshots.")
    n_cards = st.slider("Truck cards", 2, 20, 6)
    show_cameras = st.checkbox("Show cameras", value=True)


@st.cache_resource
def get_engine(url: str):
    engine = make_engine(url)
    init_schema(engine)
    return engine


def when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def plate(p) -> str:
    return (p or "—") if show_plates else mask_plate(p)


st.title("🚛 Depot cameras")


@st.fragment(run_every="2s")
def live_view():
    status = read_status(live_dir=live_dir)
    engine = get_engine(db_url)
    passes = list_passes(engine, limit=500)

    # --- status line -------------------------------------------------------------
    if status is None:
        st.warning("No pipeline status yet. Start `python scripts/run_live.py` "
                   "(or `--replay <hour folder> --speed 1`).")
        video_now = None
    else:
        age = time.time() - status.get("updated", 0)
        video_now = datetime.fromisoformat(status["video_time"]) if status.get("video_time") else None
        mode = ("🔴 LIVE" if status["mode"] == "live"
                else f"⏯️ REPLAY ×{status.get('speed') or 'max'}  ({Path(status['source']).name})")
        bits = [mode]
        if video_now:
            bits.append(f"video {video_now:%d %b %H:%M:%S}")
        if status.get("lag_s") is not None:
            bits.append(f"{status['lag_s']:.0f} s behind real time")
        bits.append(f"OCR queue {status.get('ocr_pending', 0)}")
        if status.get("stalled"):
            bits.append(f"⚠️ no data from camera {', '.join(status['stalled'])}")
        st.markdown(" · ".join(bits))
        if age > 15:
            st.error(f"The pipeline has not reported for {age:.0f} s — is it still running?")

    # --- headline numbers -----------------------------------------------------------
    ref = video_now.timestamp() if video_now else (max((p["entry_time"] for p in passes), default=0))
    hour = [p for p in passes if p["status"] == "final" and p["entry_time"] >= ref - 3600]
    in_view = [p for p in passes if p["status"] in ("provisional", "reading")]
    carriers = [p for p in hour if p["vehicle_type"] == "container_carrier" or p["codes"]]
    with_code = [p for p in carriers if p["codes"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trucks in view / being read", len(in_view))
    c2.metric("Trucks in the last hour", len(hour))
    c3.metric("Container trucks with a code", f"{len(with_code)}/{len(carriers)}" if carriers else "—")
    c4.metric("Containers seen (last hour)", sum(len(p["codes"]) for p in hour))

    # --- cameras ------------------------------------------------------------------
    if show_cameras:
        for row in (CAMERAS[:3], CAMERAS[3:]):
            cols = st.columns(3)
            for col, (cam, role) in zip(cols, row):
                data, mtime = read_camera_preview(cam, live_dir=live_dir)
                with col:
                    if data:
                        stale = time.time() - mtime > 10
                        st.image(data, caption=f"{cam} · {role}" + ("  (no new frame)" if stale else ""),
                                 width="stretch")
                    else:
                        st.caption(f"{cam} · {role}: no frame yet")

    # --- trucks in view -------------------------------------------------------------------
    st.subheader("In view now")
    if in_view:
        st.dataframe(pd.DataFrame([{
            "entered": when(p["entry_time"]), "cameras so far": p["cameras"],
            "vehicle type so far": p["vehicle_type"],
            "state": "reading codes and plates" if p["status"] == "reading" else "in view",
        } for p in sorted(in_view, key=lambda p: p["entry_time"])]), hide_index=True, width="stretch")
    else:
        st.caption("No truck at the checkpoint.")

    # --- recent trucks: profile cards ---------------------------------------------------
    st.subheader("Recent trucks")
    finals = [p for p in passes if p["status"] == "final"][:n_cards]
    if not finals:
        st.caption("No finished truck yet (a truck is final about a minute after it leaves).")
    for p in finals:
        with st.container(border=True):
            left, right = st.columns([3, 2])
            with left:
                thumbs = [(cam, thumbnail_path(p["pass_uid"], cam, live_dir=live_dir))
                          for cam, _ in CAMERAS]
                thumbs = [(cam, t) for cam, t in thumbs if t.exists()][:3]
                if thumbs:
                    tcols = st.columns(len(thumbs))
                    for tc, (cam, t) in zip(tcols, thumbs):
                        tc.image(str(t), caption=cam, width="stretch")
                else:
                    st.caption("no picture")
            with right:
                codes = " + ".join(p["codes"]) if p["codes"] else "no container code"
                st.markdown(f"**{when(p['entry_time'])}** · {p['cameras']}")
                st.markdown(f"### {codes}")
                st.markdown(f"{STATUS_MARK.get(p['code_status'], p['code_status'])}"
                            + (f" · {p['code_support']} frames" if p["codes"] else "")
                            + (f" · size/type **{p['size_type']}**" if p["size_type"] else ""))
                st.markdown(f"Vehicle: **{p['vehicle_type'].replace('_', ' ')}** "
                            f"({p['vehicle_type_conf']:.2f}) · capacity "
                            f"{CAPACITY_MARK.get(p['capacity_class'], p['capacity_class'])}")
                st.markdown(f"Plate {plate(p['plate'])} · trailer {plate(p['trailer_plate'])}"
                            + (f" · make {p['make']}" if p["make"] else ""))
                if p["notes"]:
                    with st.expander("Notes"):
                        st.write(p["notes"] if show_plates else
                                 "; ".join(n for n in p["notes"].split("; ") if not n.startswith(("plate", "trailer"))))

    # --- table ------------------------------------------------------------------------
    with st.expander(f"All trucks in the last hour ({len(hour)})"):
        if hour:
            st.dataframe(pd.DataFrame([{
                "time": when(p["entry_time"]), "codes": " ".join(p["codes"]), "status": p["code_status"],
                "size/type": p["size_type"] or "", "vehicle": p["vehicle_type"],
                "capacity": p["capacity_class"], "plate": plate(p["plate"]),
                "trailer": plate(p["trailer_plate"]), "cameras": p["cameras"],
            } for p in hour]), hide_index=True, width="stretch")


live_view()
