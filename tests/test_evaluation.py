"""Tests for the evaluation harness."""

import random
import sys
import pytest
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation import (
    EvaluationConfig,
    bootstrap_ci,
    mann_whitney_u,
    pairwise_tests,
    run_evaluation,
    summarize_cells,
)
from src.router import StrictFIFORouter, UnstrictFIFORouter


class TestBootstrapCI:
    def test_returns_three_floats(self):
        m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(m, float)
        assert lo <= m <= hi

    def test_single_value(self):
        m, lo, hi = bootstrap_ci([3.0])
        assert m == lo == hi == 3.0

    def test_ci_brackets_mean(self):
        # 1000 samples around 100 — CI should brace the true mean tightly.
        rng = random.Random(0)
        data = [100 + rng.gauss(0, 10) for _ in range(1000)]
        m, lo, hi = bootstrap_ci(data)
        assert lo < 100 < hi


class TestMannWhitney:
    def test_clearly_different_distributions(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        u, p = mann_whitney_u(a, b)
        assert p < 0.05

    def test_same_distribution(self):
        rng = random.Random(0)
        a = [rng.gauss(0, 1) for _ in range(30)]
        b = [rng.gauss(0, 1) for _ in range(30)]
        u, p = mann_whitney_u(a, b)
        assert p > 0.05    # not significant


class TestEvaluationRun:
    def test_small_grid_runs(self):
        config = EvaluationConfig(
            arrival_rates=(2.0,),
            n_seeds=3,
            duration_seconds=600.0,
            n_workers=1,    # serial for testing
        )
        routers = {
            "StrictFIFO":   lambda seed: StrictFIFORouter(),
            "UnstrictFIFO": lambda seed: UnstrictFIFORouter(alpha=0.7),
        }
        records = run_evaluation(routers, config, progress=False)
        assert len(records) == 2 * 1 * 3   # routers × rates × seeds
        for r in records:
            assert r.n_arrivals >= 0
            assert r.mean_wait >= 0

    def test_reproducible(self):
        config = EvaluationConfig(
            arrival_rates=(2.0,),
            n_seeds=3,
            duration_seconds=600.0,
            n_workers=1,
        )
        routers = {"StrictFIFO": lambda seed: StrictFIFORouter()}
        r1 = run_evaluation(routers, config, progress=False)
        r2 = run_evaluation(routers, config, progress=False)
        r1_sorted = sorted(r1, key=lambda x: x.seed)
        r2_sorted = sorted(r2, key=lambda x: x.seed)
        for a, b in zip(r1_sorted, r2_sorted):
            assert a.mean_wait == b.mean_wait
            assert a.p95_wait == b.p95_wait


class TestSummarize:
    def test_one_cell_per_router_rate(self):
        config = EvaluationConfig(
            arrival_rates=(2.0, 2.5),
            n_seeds=3,
            duration_seconds=300.0,
            n_workers=1,
        )
        routers = {
            "StrictFIFO":   lambda seed: StrictFIFORouter(),
            "UnstrictFIFO": lambda seed: UnstrictFIFORouter(alpha=0.7),
        }
        records = run_evaluation(routers, config, progress=False)
        cells = summarize_cells(records)
        assert len(cells) == 2 * 2   # 2 routers × 2 rates


class TestPairwiseTests:
    def test_returns_one_per_comparison(self):
        config = EvaluationConfig(
            arrival_rates=(2.0,),
            n_seeds=5,
            duration_seconds=300.0,
            n_workers=1,
        )
        routers = {
            "StrictFIFO":   lambda seed: StrictFIFORouter(),
            "UnstrictFIFO": lambda seed: UnstrictFIFORouter(alpha=0.7),
        }
        records = run_evaluation(routers, config, progress=False)
        tests = pairwise_tests(records, baseline_router="UnstrictFIFO")
        # 1 baseline × 1 other × 1 rate = 1 test
        assert len(tests) == 1
        assert tests[0].router_a == "UnstrictFIFO"
        assert tests[0].router_b == "StrictFIFO"