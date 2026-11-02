"""Build the footage labelling sheet (milestone M3).

The footage events are the FR1/FR3/FR4 test set, so each one needs
human-checked labels: container code, size/type, cargo configuration,
vehicle type, make, plate and fleet number. This script makes that a
correction job rather than a typing job:

- labels are pre-filled from the pipeline (fused code and size/type from
  scripts/fuse_events.py) and, optionally, from a suggestions CSV (e.g.
  by-eye notes on vehicle type, make and fleet number);
- for every event it saves large crops for checking — the cab on the
  head camera (plate, make, fleet number) and the container doors on the
  codes camera (the frame that best supports the fused code) — next to
  the existing contact sheet;
- it writes a self-contained HTML page (open it in a browser) showing
  the images beside an editable form, with a live ISO 6346 check-digit
  test on the code field. Edits autosave in the browser; "Export CSV"
  downloads labels.csv.

Pre-filled values are predictions, not ground truth: the `pred_*`
columns keep the pipeline output for comparison, and a row only counts
as labelled once its `checked` column is "yes".

If labels.csv already exists in the output folder it is used as the
starting point instead, so re-running never loses corrections.

Outputs (results/ is gitignored — plates are personal data):
    results/events/labels/labels_prefilled.csv
    results/events/labels/label_sheet.html
    results/events/labels/crops/event_NNN_{cab,doors}.jpg

Usage:
    python scripts/make_label_sheet.py
    python scripts/make_label_sheet.py --suggestions results/events/labels/claude_suggestions.csv
"""

import argparse
import csv
import html
import json
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.events import extract_events
from src.footage import load_camera_configs, load_tracks_csv, parse_recording_name

LABEL_COLUMNS = ["event", "entry_time", "cameras", "checked", "vehicle_type", "container_code",
                 "size_type", "cargo_config", "code_legible", "make", "plate", "trailer_plate",
                 "fleet_number", "notes", "pred_code", "pred_status", "pred_size_type",
                 "pred_plate", "pred_trailer_plate", "suggested"]
VEHICLE_TYPES = ["container_truck", "bobtail", "box_lorry", "light_vehicle", "other"]
CARGO_CONFIGS = ["none", "1x20", "2x20", "1x40-45", "n/a"]
LEGIBLE = ["yes", "partial", "no", "n/a"]
CROP_MAX_WIDTH = 1600


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save_crop(cap, frame_idx, box, path, pad_frac=0.10):
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)),
                 max(0, int(x1 - px)):min(w, int(x2 + px))]
    if crop.shape[1] > CROP_MAX_WIDTH:
        scale = CROP_MAX_WIDTH / crop.shape[1]
        crop = cv2.resize(crop, (CROP_MAX_WIDTH, int(crop.shape[0] * scale)))
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def save_plate_crop(cap, frame_idx, truck_box, rel_box, path, width=640):
    """Zoomed crop around a plate; rel_box is the plate inside the truck box (0..1)."""
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return False
    bx1, by1, bx2, by2 = truck_box
    bw, bh = bx2 - bx1, by2 - by1
    x1, y1, x2, y2 = (bx1 + rel_box[0] * bw, by1 + rel_box[1] * bh,
                      bx1 + rel_box[2] * bw, by1 + rel_box[3] * bh)
    px, py = (x2 - x1) * 1.2 + 20, (y2 - y1) * 1.5 + 20      # context around the plate
    h, w = frame.shape[:2]
    crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
    scale = width / crop.shape[1]
    cv2.imwrite(str(path), cv2.resize(crop, (width, int(crop.shape[0] * scale)),
                                      interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def doors_frame(event_no, fused_code, reads, detections):
    """The codes-camera detection to show: best frame reading the fused code, else largest box."""
    if fused_code:
        support = [r for r in reads if int(r["event"]) == event_no and r["code"] == fused_code]
        if support:
            best = max(support, key=lambda r: (r["status"] == "verified", float(r["mean_conf"])))
            frame = int(best["frame"])
            match = [d for d in detections if d.frame == frame]
            if match:
                return match[0]
    return max(detections, key=lambda d: d.area, default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--tracks", default="results/events/tracks.csv")
    parser.add_argument("--events-dir", default="results/events")
    parser.add_argument("--suggestions", default=None,
                        help="CSV with event + any label columns, used to pre-fill (marked 'suggested').")
    args = parser.parse_args()

    events_dir = Path(args.events_dir)
    out_dir = events_dir / "labels"
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    configs = load_camera_configs(args.config)
    videos = {info.camera: p for p in Path(args.videos_dir).glob("*.mp4")
              if (info := parse_recording_name(p.name))}
    infos = {cam: parse_recording_name(p.name) for cam, p in videos.items()}
    ex = extract_events(load_tracks_csv(args.tracks), configs, infos)
    cab_cam = next(c for c in ex.chain if configs[c].role == "cab")
    codes_cam = next(c for c in ex.chain if configs[c].role == "codes")

    events = {int(r["event"]): r for r in read_csv(events_dir / "events.csv")}
    fused = {int(r["event"]): r for r in read_csv(events_dir / "fused.csv")}
    plates = ({int(r["event"]): r for r in read_csv(events_dir / "plates.csv")}
              if (events_dir / "plates.csv").exists() else {})
    reads = [r for r in read_csv(events_dir / "reads.csv") if r["camera"] == codes_cam]
    suggestions = ({int(r["event"]): r for r in read_csv(args.suggestions)}
                   if args.suggestions else {})
    existing_path = out_dir / "labels.csv"
    existing = ({int(r["event"]): r for r in read_csv(existing_path)}
                if existing_path.exists() else {})
    if len(ex.events) != len(events):
        sys.exit(f"events.csv has {len(events)} events but tracks give {len(ex.events)}; "
                 "re-run extract_events.py and fuse_events.py first")

    import cv2

    caps = {cam: cv2.VideoCapture(str(videos[cam])) for cam in ex.chain}
    truck_boxes = {(cam, d.frame): d.box for (cam, _), dets in ex.detections.items() for d in dets}
    rows, images = [], {}
    for i, e in enumerate(ex.events, 1):
        ev, fu, pl = events[i], fused.get(i, {}), plates.get(i, {})
        row = {c: "" for c in LABEL_COLUMNS}
        row.update(event=i, entry_time=ev["entry_time"], cameras=ev["cameras"], checked="no",
                   container_code=fu.get("all_code", ""), size_type=fu.get("all_size_type", ""),
                   pred_code=fu.get("all_code", ""), pred_status=fu.get("all_status", ""),
                   pred_size_type=fu.get("all_size_type", ""),
                   plate=pl.get("vehicle_plate", ""), trailer_plate=pl.get("trailer_plate", ""),
                   pred_plate=pl.get("vehicle_plate", ""),
                   pred_trailer_plate=pl.get("trailer_plate", ""))
        suggested = []
        for col, value in suggestions.get(i, {}).items():
            if col in LABEL_COLUMNS and col != "event" and value:
                row[col] = value
                suggested.append(col)
        row["suggested"] = " ".join(suggested)
        if i in existing:
            old = existing[i]
            prefilled = {c: row[c] for c in LABEL_COLUMNS}
            # Earlier labels win over everything...
            row.update({c: old[c] for c in LABEL_COLUMNS if c in old})
            # ...except that a blank label is not a correction: fill it from the new
            # prediction, shade it as a suggestion, and ask for the row to be re-checked.
            filled = [c for c in LABEL_COLUMNS
                      if not c.startswith("pred_") and c not in ("checked", "suggested", "notes")
                      and not old.get(c) and prefilled[c]]
            for c in filled:
                row[c] = prefilled[c]
            if filled:
                row["checked"] = "no"
                row["suggested"] = " ".join(sorted(set(row["suggested"].split()) | set(filled)))
            row.update({c: prefilled[c] for c in LABEL_COLUMNS if c.startswith("pred_")})
        rows.append(row)

        images[i] = []
        sheet = events_dir / "sheets" / f"event_{i:03d}.jpg"
        if sheet.exists():
            images[i].append(("contact sheet", f"../sheets/{sheet.name}"))
        cab_dets = [d for p in e.on(cab_cam) for d in ex.detections[(cab_cam, p.track_id)]]
        if cab_dets:
            # Largest cab box = truck closest to the camera: plate and fleet number biggest.
            d = max(cab_dets, key=lambda d: d.area)
            name = f"event_{i:03d}_cab.jpg"
            if save_crop(caps[cab_cam], d.frame, d.box, out_dir / "crops" / name):
                images[i].append((f"cab — camera {cab_cam}, frame {d.frame}", f"crops/{name}"))
        code_dets = [d for p in e.on(codes_cam) for d in ex.detections[(codes_cam, p.track_id)]]
        d = doors_frame(i, fu.get("all_code"), reads, code_dets)
        if d is not None:
            name = f"event_{i:03d}_doors.jpg"
            if save_crop(caps[codes_cam], d.frame, d.box, out_dir / "crops" / name):
                images[i].append((f"rear / doors — camera {codes_cam}, frame {d.frame}",
                                  f"crops/{name}"))
        # Zoomed plate crops from the frame where each fused plate was read best.
        for kind in ("vehicle", "trailer"):
            evidence = pl.get(f"{kind}_evidence", "")
            if not evidence:
                continue
            cam, frame_idx, box = evidence.split(":")
            frame_idx, rel = int(frame_idx), tuple(float(v) for v in box.split(","))
            if (cam, frame_idx) not in truck_boxes:
                continue
            name = f"event_{i:03d}_{kind}_plate.jpg"
            if save_plate_crop(caps[cam], frame_idx, truck_boxes[(cam, frame_idx)], rel,
                               out_dir / "crops" / name):
                images[i].append((f"{kind} plate as read ({pl[f'{kind}_plate']}, "
                                  f"{pl[f'{kind}_support']} frames) — camera {cam}, frame {frame_idx}",
                                  f"crops/{name}"))
    for cap in caps.values():
        cap.release()

    with (out_dir / "labels_prefilled.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "label_sheet.html").write_text(render_html(rows, images))
    print(f"{len(rows)} events -> {out_dir / 'labels_prefilled.csv'}, "
          f"{out_dir / 'label_sheet.html'} and crops/")
    if existing:
        print(f"Started from existing {existing_path} ({len(existing)} rows)")


def render_html(rows, images) -> str:
    data = json.dumps({"rows": rows, "images": {str(k): v for k, v in images.items()},
                       "columns": LABEL_COLUMNS, "vehicleTypes": VEHICLE_TYPES,
                       "cargoConfigs": CARGO_CONFIGS, "legible": LEGIBLE})
    return HTML_TEMPLATE.replace("/*DATA*/null", data.replace("</", "<\\/"))


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Footage Labels</title>
<style>
  :root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#deded8;
          --ok:#1f7a4d; --bad:#b3261e; --warn:#8a5a00; --accent:#2a5bd7; --sugg:#fff6db; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --card:#20201e; --ink:#ececea; --muted:#9a9a94; --line:#33332f;
            --ok:#5cc28d; --bad:#f2847b; --warn:#e6b85c; --accent:#7ea2ff; --sugg:#3a3320; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line);
           padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0 12px 0 0; }
  .progress { color:var(--muted); }
  .spacer { flex:1; }
  button { font:inherit; padding:6px 12px; border-radius:6px; border:1px solid var(--line);
           background:var(--card); color:var(--ink); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  main { max-width:1500px; margin:0 auto; padding:16px; }
  .event { background:var(--card); border:1px solid var(--line); border-radius:10px;
           margin-bottom:20px; display:grid; grid-template-columns:minmax(0,1fr) 380px; overflow:hidden; }
  .event.done { border-color:var(--ok); }
  .imgs { padding:12px; display:flex; flex-direction:column; gap:10px; min-width:0; }
  .imgs figure { margin:0; }
  .imgs img { max-width:100%; display:block; border-radius:6px; cursor:zoom-in; }
  .imgs figcaption { color:var(--muted); font-size:12px; margin-top:2px; }
  form { padding:12px 14px; border-left:1px solid var(--line); display:flex; flex-direction:column; gap:8px; }
  form h2 { font-size:15px; margin:0; }
  .meta { color:var(--muted); font-size:12px; margin-top:-4px; }
  label { display:grid; grid-template-columns:110px 1fr; align-items:center; gap:8px; font-size:13px; }
  input, select, textarea { font:inherit; padding:5px 7px; border-radius:5px; border:1px solid var(--line);
                            background:var(--bg); color:var(--ink); width:100%; min-width:0; }
  textarea { min-height:48px; resize:vertical; }
  .sugg { background:var(--sugg); }
  .hint { font-size:12px; grid-column:2; margin-top:-6px; }
  .hint.ok { color:var(--ok); } .hint.bad { color:var(--bad); } .hint.warn { color:var(--warn); }
  .pred { font-size:12px; color:var(--muted); border-top:1px dashed var(--line); padding-top:6px; }
  .checked { display:flex; gap:8px; align-items:center; font-weight:600; }
  .checked input { width:auto; }
  .legend { font-size:12px; color:var(--muted); }
  .legend span { background:var(--sugg); padding:0 4px; border-radius:3px; }
  dialog { border:none; padding:0; max-width:96vw; max-height:96vh; background:transparent; }
  dialog img { max-width:96vw; max-height:96vh; display:block; }
  dialog::backdrop { background:rgba(0,0,0,.8); }
  @media (max-width: 900px) {
    .event { grid-template-columns:1fr; }
    form { border-left:none; border-top:1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <h1>Footage labels</h1>
  <span class="progress" id="progress"></span>
  <span class="legend"><span>shaded</span> = pre-filled suggestion, please check</span>
  <span class="spacer"></span>
  <button id="import">Import CSV</button>
  <button class="primary" id="export">Export CSV</button>
  <input type="file" id="file" accept=".csv" hidden>
</header>
<main id="events"></main>
<dialog id="zoom"><img alt=""></dialog>
<script>
const DATA = /*DATA*/null;
// Bump when the sheet is regenerated with new columns or pre-fills, so a
// browser's autosave of an older sheet cannot override them (labels.csv,
// which the sheet starts from, already holds every exported correction).
const KEY = "depot-footage-labels-v2";
const COLS = DATA.columns;

function load() {
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) {}
  if (!saved) return DATA.rows.map(r => ({...r}));
  const byEvent = Object.fromEntries(saved.map(r => [String(r.event), r]));
  return DATA.rows.map(r => ({...r, ...(byEvent[String(r.event)] || {})}));
}
let rows = load();
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(rows)); } catch (e) {}
  updateProgress();
}

// ISO 6346 check digit (Appendix A of the brief).
function letterValue(ch) {
  let v = 10;
  for (let c = 65; c <= 90; c++) {
    if (v % 11 === 0) v++;
    if (String.fromCharCode(c) === ch) return v;
    v++;
  }
  return null;
}
function oneCodeHint(code) {
  if (!/^[A-Z]{3}[UJZ][0-9]{7}$/.test(code)) return ["bad", `${code}: format is 3 letters + U/J/Z + 7 digits`];
  let sum = 0;
  for (let i = 0; i < 10; i++) {
    const ch = code[i];
    const v = /[0-9]/.test(ch) ? Number(ch) : letterValue(ch);
    sum += v * Math.pow(2, i);
  }
  const check = (sum % 11) % 10;
  return Number(code[10]) === check
    ? ["ok", `${code}: check digit valid`]
    : ["warn", `${code}: check digit should be ${check} — typo, or printed wrong on the box?`];
}
// One code, or two separated by a space for a 2x20 ft truck.
function codeHint(field) {
  const codes = (field || "").toUpperCase().split(/\\s+/).filter(Boolean);
  if (!codes.length) return ["", ""];
  if (codes.length > 2) return ["bad", "at most two codes (2x20 ft)"];
  const hints = codes.map(oneCodeHint);
  const worst = ["bad", "warn", "ok"].find(level => hints.some(([c]) => c === level));
  return [worst, hints.map(([, t]) => t).join(" · ")];
}

function field(row, col, kind, options) {
  const sugg = (row.suggested || "").split(" ").includes(col);
  const id = `f-${row.event}-${col}`;
  let input;
  if (kind === "select") {
    const opts = [""].concat(options).map(o =>
      `<option value="${o}" ${row[col] === o ? "selected" : ""}>${o || "—"}</option>`).join("");
    input = `<select id="${id}" data-col="${col}" class="${sugg ? "sugg" : ""}">${opts}</select>`;
  } else if (kind === "textarea") {
    input = `<textarea id="${id}" data-col="${col}" class="${sugg ? "sugg" : ""}">${esc(row[col])}</textarea>`;
  } else {
    input = `<input id="${id}" data-col="${col}" value="${esc(row[col])}" class="${sugg ? "sugg" : ""}" autocomplete="off">`;
  }
  return `<label for="${id}">${col.replace(/_/g, " ")}${input}</label>`;
}
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

function render() {
  const main = document.getElementById("events");
  main.innerHTML = rows.map(row => {
    const imgs = (DATA.images[String(row.event)] || []).map(([cap, src]) =>
      `<figure><img src="${src}" alt="${esc(cap)}" loading="lazy"><figcaption>${esc(cap)}</figcaption></figure>`).join("");
    return `<section class="event ${row.checked === "yes" ? "done" : ""}" data-event="${row.event}">
      <div class="imgs">${imgs || "<p>No images.</p>"}</div>
      <form autocomplete="off">
        <h2>Event ${row.event}</h2>
        <div class="meta">${esc(row.entry_time)} · cameras ${esc(row.cameras)}</div>
        ${field(row, "vehicle_type", "select", DATA.vehicleTypes)}
        ${field(row, "container_code", "text")}
        <div class="hint" data-hint="code"></div>
        ${field(row, "size_type", "text")}
        ${field(row, "cargo_config", "select", DATA.cargoConfigs)}
        ${field(row, "code_legible", "select", DATA.legible)}
        ${field(row, "make", "text")}
        ${field(row, "plate", "text")}
        ${field(row, "trailer_plate", "text")}
        ${field(row, "fleet_number", "text")}
        ${field(row, "notes", "textarea")}
        <div class="pred">Pipeline: ${esc(row.pred_code) || "no code"} ${row.pred_status ? "(" + esc(row.pred_status) + ")" : ""}
          ${row.pred_size_type ? "· " + esc(row.pred_size_type) : ""}
          · plate ${esc(row.pred_plate) || "–"} · trailer ${esc(row.pred_trailer_plate) || "–"}</div>
        <label class="checked"><input type="checkbox" data-col="checked" ${row.checked === "yes" ? "checked" : ""}>
          I have checked every field of this event</label>
      </form></section>`;
  }).join("");
  main.querySelectorAll("section").forEach(sec => updateHint(sec));
  updateProgress();
}
function updateHint(sec) {
  const row = rows.find(r => String(r.event) === sec.dataset.event);
  const [cls, text] = codeHint(row.container_code);
  const hint = sec.querySelector('[data-hint="code"]');
  hint.className = "hint " + cls;
  hint.textContent = text;
}
function updateProgress() {
  const n = rows.filter(r => r.checked === "yes").length;
  document.getElementById("progress").textContent = `${n} / ${rows.length} checked`;
}

document.getElementById("events").addEventListener("input", ev => {
  const el = ev.target, col = el.dataset.col;
  if (!col) return;
  const sec = el.closest("section");
  const row = rows.find(r => String(r.event) === sec.dataset.event);
  if (col === "checked") {
    row.checked = el.checked ? "yes" : "no";
    sec.classList.toggle("done", el.checked);
  } else {
    row[col] = ["container_code", "size_type", "plate", "trailer_plate"].includes(col)
      ? el.value.toUpperCase().trim().replace(/\\s+/g, " ") : el.value;
    el.classList.remove("sugg");
    row.suggested = (row.suggested || "").split(" ").filter(c => c && c !== col).join(" ");
    if (col === "container_code") updateHint(sec);
  }
  save();
});
document.getElementById("events").addEventListener("click", ev => {
  if (ev.target.tagName !== "IMG") return;
  const dlg = document.getElementById("zoom");
  dlg.querySelector("img").src = ev.target.src;
  dlg.showModal();
});
document.getElementById("zoom").addEventListener("click", ev => ev.currentTarget.close());

function toCsv() {
  const q = v => { v = String(v ?? ""); return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  return [COLS.join(",")].concat(rows.map(r => COLS.map(c => q(r[c])).join(","))).join("\\n") + "\\n";
}
document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([toCsv()], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});
function parseCsv(text) {
  const out = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') q = false;
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cur); cur = ""; }
    else if (ch === "\\n" || ch === "\\r") {
      if (ch === "\\r" && text[i + 1] === "\\n") i++;
      row.push(cur); cur = ""; if (row.some(v => v !== "")) out.push(row); row = [];
    } else cur += ch;
  }
  if (cur !== "" || row.length) { row.push(cur); out.push(row); }
  const [head, ...body] = out;
  return body.map(r => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""])));
}
document.getElementById("import").addEventListener("click", () => document.getElementById("file").click());
document.getElementById("file").addEventListener("change", async ev => {
  const file = ev.target.files[0];
  if (!file) return;
  const byEvent = Object.fromEntries(parseCsv(await file.text()).map(r => [String(r.event), r]));
  rows = rows.map(r => ({...r, ...(byEvent[String(r.event)] || {})}));
  save(); render();
});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
