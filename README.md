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
2. **Unstrict FIFO routing.** A gate-allocation algorithm that scores
   gates by a weighted combination of recent throughput and queue length,
   evaluated against four baselines (strict FIFO, round-robin, random,
   shortest-queue) across 30 random seeds and five arrival rates.
3. **Unified identification architecture.** Modular pipeline with
   placeholders for vehicle make/model recognition and license plate
   recognition (FYP2 work), connected through a profile fusion layer.

## Repository layout

```
DepotManagementSystemFYP/
├── src/
│   ├── iso6346.py            # ISO 6346 validation and distribution-guided recovery
│   ├── iso6346_recovery.py   # OCR-confusion substitution recovery (used by real OCR)
│   ├── db.py                 # SQLAlchemy schema and queries (SQLite, PostgreSQL-ready)
│   ├── ocr.py                # OCRAdapter ABC, MockOCRAdapter, RealOCRAdapter
│   ├── image_source.py       # ImageSource ABC, SyntheticImageSource, RealImageSource
│   ├── synthetic_images.py   # Procedural container code image generator
│   ├── router.py             # Router ABC, UnstrictFIFORouter + 4 baselines
│   ├── simulation.py         # SimPy discrete-event simulation harness
│   ├── evaluation.py         # Multi-seed evaluation with bootstrap CIs
│   ├── dashboard_io.py       # Atomic frame writes for the live dashboard
│   └── CCLN.py               # Container code localization network (model definition)
├── scripts/
│   ├── run_evaluation.py     # Multi-seed router comparison (5 routers × 5 rates × 30 seeds)
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

All ~60 tests should pass. The router and evaluation tests are the slowest
(~10s combined); everything else completes in well under a second.

### Run the routing algorithm evaluation

This is the multi-seed statistical comparison used for the report. Takes
~3 minutes on 16 cores.

```bash
python scripts/run_evaluation.py
```

Produces:
- `results/results.csv` — one row per (router, arrival_rate, seed)
- `results/statistics.json` — bootstrap CIs and pairwise tests

To regenerate the plots and report-ready tables from the CSV:

```bash
python scripts/plot_evaluation.py
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

The unstrict FIFO algorithm exposes one knob: `alpha` (default 0.7),
controlling the throughput-vs-queue tradeoff. See `src/router.py` docstring
for details.

## Limitations


1. **The queue-balancing term in unstrict FIFO is structurally inactive
   under the global-queue model used here.** All trucks queue in one global
   FIFO; gates are assigned at dispatch time. The score formula's
   queue-length penalty (`-(1-α) · Q_g`) is therefore always zero, and the
   algorithm reduces to throughput-greedy gate selection. A per-gate-queue
   deployment would activate the full algorithm — recommended for FYP2.
2. **Wait-time improvements are modest** (7%–36% reduction over strict
   FIFO across tested rates), statistically significant only at low load
   (1.5/min, p=0.006 Bonferroni-corrected). The operational story is the
   robust load redistribution: Gate A utilization drops ~20pp, Gate C
   rises ~12pp.
3. **The OCR pipeline is ~80% accurate** on the current real-image set
   without fine-tuning. PaddleOCR was used out of the box; fine-tuning on
   the project's labeled container dataset is a recommended next step.
4. **Synthetic images are for pipeline testing only.** They do not match
   the training distribution of CCLN and would not pass through the
   localization stage reliably. The dashboard demo uses real container
   photos from `images/`.

## Roadmap (FYP2)

- Vehicle make/model recognition (ResNet-50 / Stanford Cars)
- License plate recognition (YOLOv8 + OCR)
- OCR accepting a real-time video input instead of images
- Profile fusion combining all three identification outputs
- Per-gate queues + algorithm refinement
- PostgreSQL migration
- Fine-tuned OCR on labeled dataset
- Production dashboard with operator controls