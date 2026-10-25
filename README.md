# Depot Traffic Management System

Final Year Project 1 — University of Malaya
Author: Muhammad Aiman Bin Sharuddin (23063615)
Supervisor: Dr. Liew Wei Shung
Collaborator: Innonics Sdn. Bhd.

A depot traffic management system combining computer-vision container code
recognition with throughput-aware gate routing to reduce congestion at
container depot entry points.

## What this project does

A simulated depot accepts arriving trucks, identifies each truck's container
code from a camera image, and routes the truck to one of three gates using
a routing algorithm that adapts to per-gate throughput in real time. The
system runs as a discrete-event simulation for algorithm evaluation, and
as a real-time process with a live dashboard for operator-facing demos.

Three research objectives addressed:

1. **OCR pipeline with ISO 6346 validation and recovery.** A two-stage
   container code recognition pipeline (CCLN localization + PaddleOCR
   transcription) integrated with ISO 6346 check-digit verification.
   Recovery uses OCR-confusion-aware character substitution for invalid
   reads (e.g. O↔0, S↔5).
2. **Unstrict FIFO routing.** A gate-allocation algorithm that keeps trucks
   in arrival order but lets the head truck skip a free slow gate when a
   busy fast gate is expected to clear it sooner (v2, FYP2). It plans the
   queue against each gate's observed service time and the recent arrival
   rate. Evaluated against the FYP1 version and three baselines (strict
   FIFO, round-robin, random) across 30 random seeds and five arrival rates.
3. **Unified identification architecture.** Modular pipeline with
   placeholders for vehicle make/model recognition and license plate
   recognition (FYP2 work), connected through a profile fusion layer.

## Repository layout

```
DepotManagementSystemFYP/
├── src/
│   ├── iso6346.py            # ISO 6346 validation and distribution-guided recovery
│   ├── iso6346_recovery.py   # Reads codes from OCR fragments; verified/recovered/inferred status
│   ├── db.py                 # SQLAlchemy schema and queries (SQLite, PostgreSQL-ready)
│   ├── ocr.py                # OCRAdapter ABC, MockOCRAdapter, RealOCRAdapter
│   ├── image_source.py       # ImageSource ABC, SyntheticImageSource, RealImageSource
│   ├── synthetic_images.py   # Procedural container code image generator
│   ├── router.py             # Router ABC, UnstrictFIFOv2Router, UnstrictFIFORouter + baselines
│   ├── simulation.py         # SimPy discrete-event simulation harness
│   ├── evaluation.py         # Multi-seed evaluation with bootstrap CIs
│   ├── dashboard_io.py       # Atomic frame writes for the live dashboard
│   └── CCLN.py               # Container code localization network (model definition)
├── scripts/
│   ├── run_evaluation.py     # Multi-seed router comparison (5 routers × 5 rates × 30 seeds)
│   ├── tune_unstrict_v2.py   # Lookahead sensitivity sweep for Unstrict FIFO v2 (held-out seeds)
│   ├── robustness_checks.py  # v2 under milder gate speeds and high service-time variance
│   ├── run_realtime_sim.py   # Real-time simulator for the dashboard demo
│   ├── dashboard.py          # Streamlit operator dashboard
│   └── plot_evaluation.py    # Generate report-ready plots and tables from results.csv
├── tests/                    # pytest suite, one file per module
├── models/
│   └── ccln.pth              # Pre-trained CCLN weights (gitignored, if you want the file email me at aimansha887@gmail.com)
├── images/                   # Real container photos (filename = ISO 6346 code)
├── results/                  # Evaluation outputs (CSV, JSON, plots)
└── frames/                   # Live frame buffer for the dashboard (gitignored)
```

## Getting started

### Install

```bash
git clone https://github.com/Sishdoap/DepotManagementSystemFYP.git
cd DepotManagementSystemFYP
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the tests

```bash
pytest tests/ -v
```

All ~250 tests should pass in about 25 s; the simulation and evaluation
tests are the slowest. None of them need the OCR models.

### Run the routing algorithm evaluation

This is the multi-seed statistical comparison used for the report. Takes
~4 minutes on 16 cores.

```bash
python scripts/run_evaluation.py
```

Produces:
- `results/results.csv` — one row per (router, arrival_rate, seed)
- `results/statistics.json` — bootstrap CIs and pairwise tests

The primary metric is **time in system** (arrival to departure). Queue wait
(arrival to gate assignment) is reported too, but on its own it rewards
sending a truck straight to a slow gate. The first 10 simulated minutes of
each 2-hour run are discarded as warm-up.

To re-run the sensitivity sweep that chose Unstrict FIFO v2's default
lookahead (uses seeds 1000–1029, separate from the evaluation's 0–29):

```bash
python scripts/tune_unstrict_v2.py
```

To check how the result depends on the gate assumptions — milder gate speed
differences (the live-demo config) and high service-time variance — run the
robustness checks (~5 minutes; writes `results/robustness.csv` and
`results/robustness.md`):

```bash
python scripts/robustness_checks.py
```

To regenerate the plots and report-ready tables from the CSV:

```bash
python scripts/plot_evaluation.py
```

### Run the OCR benchmark

Runs CCLN + PaddleOCR on every labelled image in `images/` (~1 hour on CPU
with `--all-rotations`) and saves each image's OCR fragments for all four
rotations:

```bash
python scripts/benchmark_ocr.py --all-rotations --csv-output results/ocr_capture_all_rotations.csv
```

The code-reading step can then be re-evaluated offline in seconds, without
re-running OCR. This compares the FYP1 recovery with each configuration of
the current reader and writes `results/recovery_ablation.md`:

```bash
python scripts/replay_recovery.py
```

### Run the live demo (dashboard)

Two processes in two terminals.

**Terminal 1 — the simulator:**
```bash
python scripts/run_realtime_sim.py
```

This loads the CCLN + PaddleOCR pipeline (~15s warmup) and runs the
simulator in real-time mode with `REALTIME_FACTOR=0.2` (5× faster than
real life). Truck arrival rate is 2/min by default.

**Terminal 2 — the dashboard:**
```bash
streamlit run scripts/dashboard.py
```

Opens at http://localhost:8501. Shows:
- Waiting queue
- Per-gate camera feeds with CCLN bounding boxes drawn
- Recent container code reads (raw OCR vs final recovered code)
- Per-gate throughput chart over the last 30 simulated minutes

Stop both with Ctrl-C.

## Key design decisions

- **Modular components behind clean interfaces.** Each piece (OCR adapter,
  router, image source) is an abstract base class with multiple
  implementations. Swapping the real OCR for the mock, or one router for
  another, requires no changes to the surrounding code.
- **Reproducibility.** Every random operation accepts an explicit
  `random.Random` instance. Given a fixed seed, the simulation produces
  bit-identical results.
- **SQLite for FYP1, PostgreSQL-ready for FYP2.** Connection URL is the
  only thing that changes between backends; queries use SQLAlchemy Core.
- **Two simulation modes share one core.** Fast mode (no real-time delay)
  for evaluation; real-time mode (wall-clock paced) for the dashboard.
  Same simulator class, same routers, same database.

## Configuration

The simulator's defaults are in `src/simulation.py:SimulationConfig`:
- `arrival_rate_per_minute=1.5`
- `gates=[A=90s, B=60s, C=45s]`
- `throughput_window_seconds=600`
- `seed=0`

The evaluation uses heavier gate heterogeneity (A=180s, B=90s, C=30s) to
test routing under realistic depot conditions where lane speeds differ
significantly.

Unstrict FIFO v2 exposes one knob: `lookahead_seconds` (default 90), the
window of expected arrivals added to the queue when planning. Smaller
values hold trucks for fast gates more eagerly; larger values use slow
gates sooner. 60–120 s is a flat optimum in the sensitivity sweep. The FYP1
router's knob is `alpha` (default 0.7). See the `src/router.py` docstrings.

## Limitations


1. **The >25% target is met only at low load.** Unstrict FIFO v2 cuts mean
   time in system versus strict FIFO by 26.4% at 1.5 trucks/min, then
   19.3%, 14.7%, 10.6% and 6.4% at 1.8–2.4/min (Bonferroni-corrected
   Mann-Whitney: significant up to 2.2/min, not at 2.4/min). This is
   expected: holding a truck for a fast gate only pays when there is spare
   fast-gate capacity, and as load approaches the depot's total capacity
   (3 trucks/min in the evaluation config) every policy must use every
   gate. v2 beats the FYP1 router significantly up to 2.0/min; above that
   the two are indistinguishable. **The gain also depends on how different
   the gates are:** with milder speed differences (90/60/45 s) it falls to
   2–7%, and with highly variable service times the high-load advantage
   disappears. v2 is never significantly worse on time in system in any
   tested scenario (`scripts/robustness_checks.py`).
2. **v2 trades queue wait for time in system.** Because it may hold a
   truck for a faster gate, its mean queue wait is slightly *higher* than
   strict FIFO's (e.g. 10.1 s vs 7.8 s at 1.5/min) while total time in
   system is much lower. Judge routing on time in system; FYP1's results
   reported queue wait only.
3. **Global queue, not per-gate lanes.** All trucks share one FIFO queue
   and gates are chosen at dispatch time. This is the theoretically better
   arrangement (no truck is stuck in the wrong lane), but it means
   ShortestQueue is meaningless here and the FYP1 router's queue-length
   term is always zero. v2 uses the queue length in its plan instead.
4. **Results before FYP2 were affected by a dispatcher bug** that could
   assign several trucks to one gate in a single dispatch step. FYP1's
   published numbers should not be compared directly with the current ones.
5. **OCR is not yet fine-tuned.** On all 1,386 labelled images (labels
   audited against the check digit; see `images/RELABEL_LOG.csv`) the
   pipeline validates 74.8% of codes correctly, with 9 codes (0.6%)
   validated but wrong — mostly PaddleOCR reading "8" as "0" in a way that
   still passes the check digit. Another 18% are *inferred*: the boxed
   check digit was missing or misread and was computed from the other ten
   characters (80% of these are correct, but none are verified). PaddleOCR
   is used out of the box; fine-tuning on the labelled crops is the next
   step. See `results/recovery_ablation.md`.
6. **Synthetic images are for pipeline testing only.** They do not match
   the training distribution of CCLN and would not pass through the
   localization stage reliably. The dashboard demo uses real container
   photos from `images/`.

## Roadmap (FYP2)

- Vehicle make/model recognition (ResNet-50 / Stanford Cars)
- License plate recognition (YOLOv8 + OCR)
- OCR accepting a real-time video input instead of images
- Profile fusion combining all three identification outputs
- Load-aware routing (service time predicted from the truck's capacity class)
- PostgreSQL migration
- Fine-tuned OCR on labeled dataset
- Production dashboard with operator controls