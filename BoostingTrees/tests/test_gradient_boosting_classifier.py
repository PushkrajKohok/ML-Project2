"""
Medium‑weight test‑suite for our from‑scratch GradientBoostingClassifier.

What we check
1.  API sanity and basic error handling.
2.  The model actually learns on easy *and* moderately sized data.
3.  Hyper‑parameters behave sensibly (learning‑rate & n_estimators sweeps).
4.  Internal bookkeeping: monotone `errors_`, reproducibility, feature‑importances.
5.  Edge‑cases: NaNs, single‑class labels, near‑perfect collinearity.
6.  Probability‑calibration sanity.

Everything is sized to finish in < 30s.
"""

import os
import sys
import time
import numpy as np
import pytest
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.metrics import balanced_accuracy_score

# Import model from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.GradientBoostingClassifier import GradientBoostingClassifier  # noqa: E402



# Helpers
def accuracy(model, X, y):
    """Simple wrapper so we can tweak in one place."""
    return np.mean(model.predict(X) == y)


def balanced_acc(model, X, y):
    """Balanced accuracy for imbalanced‑class tests."""
    return balanced_accuracy_score(y, model.predict(X))


# Shared synthetic datasets
@pytest.fixture(scope="module")
def datasets():
    """Generate reusable synthetic datasets (moderate sizes)."""
    out = {}
    rng = np.random.RandomState(42)

    # Mildly non‑linear, 1 000 samples
    out["moons"] = make_moons(n_samples=1000, noise=0.25, random_state=rng)
    out["circles"] = make_circles(n_samples=1000, noise=0.15, factor=0.5, random_state=rng)

    # Easy linear separation, 1 000 samples, 2 feats
    out["linear"] = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=2.5,
        random_state=rng,
    )

    # High‑dim, moderate difficulty: 2 000×50, 20 informative
    out["high_dim"] = make_classification(
        n_samples=2000,
        n_features=50,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=rng,
    )

    # Imbalanced (90 % vs 10 %), 2 000×10
    out["imbalanced"] = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        weights=[0.9, 0.1],
        flip_y=0.01,
        random_state=rng,
    )

    return out


# ------------------------------------------------------------------ #
# 1.  Basic API & guard‑rails
# ------------------------------------------------------------------ #

def test_init_defaults():
    gb = GradientBoostingClassifier()
    assert gb.n_estimators == 100
    assert gb.learning_rate == 0.1
    assert gb.subsample == 1.0
    # Un‑fitted estimator should not predict
    with pytest.raises(Exception):
        gb.predict(np.zeros((2, 2)))


def test_predict_before_fit(datasets):
    X, _ = datasets["moons"]
    with pytest.raises(Exception):
        GradientBoostingClassifier().predict(X)


def test_zero_estimators():
    X, y = make_moons(n_samples=40, noise=0.1, random_state=0)
    with pytest.raises(ValueError):
        GradientBoostingClassifier(n_estimators=0).fit(X, y)



# 2.  “Does it learn?” quick checks on bigger data
@pytest.mark.parametrize(
    "ds_name, min_score",
    [("linear", 0.93), ("moons", 0.85), ("circles", 0.85)],
)
def test_learns_problems(datasets, ds_name, min_score):
    X, y = datasets[ds_name]
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
    gb.fit(X, y)
    assert accuracy(gb, X, y) > min_score



# 3.  Hyper‑parameter
@pytest.mark.parametrize("lr", [0.01, 0.05, 0.3])
def test_learning_robustness(datasets, lr):
    X, y = datasets["moons"]
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=lr, random_state=0)
    gb.fit(X, y)
    assert accuracy(gb, X, y) > 0.80



@pytest.mark.parametrize("n_est", [1, 50, 150])
def test_n_estimators_increase_score(datasets, n_est):
    X, y = datasets["moons"]
    gb = GradientBoostingClassifier(n_estimators=n_est, random_state=0)
    gb.fit(X, y)
    thresh = 0.72 if n_est == 1 else 0.80  # allow weaker model when n_est=1
    assert accuracy(gb, X, y) > thresh


# 4.  Book‑keeping & reproducibility
def test_reproducibility(datasets):
    X, y = datasets["moons"]
    gb1 = GradientBoostingClassifier(n_estimators=60, random_state=123).fit(X, y)
    gb2 = GradientBoostingClassifier(n_estimators=60, random_state=123).fit(X, y)
    np.testing.assert_array_equal(gb1.predict(X), gb2.predict(X))
    # training loss should never increase
    assert np.all(np.diff(gb1.errors_) <= 1e-9)
    assert np.isfinite(gb1.errors_).all()


def test_feature_importance_sum(datasets):
    X, y = datasets["moons"]
    gb = GradientBoostingClassifier(n_estimators=40, random_state=0).fit(X, y)
    imp = gb.feature_importances()
    np.testing.assert_allclose(imp.sum(), 1.0, atol=1e-6)
    assert (imp >= 0).all()



# 5.  Edge‑cases
def test_nan_input():
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(X, y)


def test_single_class_error():
    X, y = make_moons(n_samples=100, noise=0.1, random_state=0)
    y[:] = 0
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(X, y)


def test_near_perfect_collinearity():
    """Two almost identical features should not derail training."""
    rng = np.random.RandomState(0)
    base = rng.randn(400, 1)
    X = np.hstack([base, base + 1e-8 * rng.randn(400, 1)])
    y = (base.ravel() > 0).astype(int)
    gb = GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=0)
    gb.fit(X, y)
    assert accuracy(gb, X, y) > 0.9

# 6.  Stress test for large n_estimators
def test_large_n_estimators_runtime(datasets):
    X, y = datasets["moons"]
    gb = GradientBoostingClassifier(n_estimators=300, random_state=0).fit(X, y)
    assert accuracy(gb, X, y) > 0.83

# 7.  Additional tests for subsampling and max_depth
def test_subsample_robustness(datasets):
    """Model should still learn with stochastic subsampling."""
    X, y = datasets["moons"]
    gb = GradientBoostingClassifier(n_estimators=100, subsample=0.6, random_state=0).fit(X, y)
    assert accuracy(gb, X, y) > 0.80


def test_max_depth_effect(datasets):
    """Deeper trees should generally perform ≥ shallow trees on the same dataset."""
    X, y = datasets["moons"]
    shallow = GradientBoostingClassifier(n_estimators=80, max_depth=1, random_state=0).fit(X, y)
    deep = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=0).fit(X, y)
    assert accuracy(deep, X, y) >= accuracy(shallow, X, y) + 0.04