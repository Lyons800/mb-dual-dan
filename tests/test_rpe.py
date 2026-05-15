"""Phase 1 regression tests for the Bennett RPE baseline.

Pin down the acquisition behaviour so any future refactor surfaces a regression.
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair


def _train(n_trials=60, seed=0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    model = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    cs = make_cs_pair(model.coder.n_pn, active_frac=0.1, seed=seed + 1)

    history = {"label": [], "m_hat": [], "dan": []}
    for odor, reward, label in conditioning_trials(cs, n_trials):
        out = model.step(odor, reward)
        history["label"].append(label)
        history["m_hat"].append(out["m_hat"])
        history["dan"].append(out["dan"])
    return model, history


def test_kc_sparsity_targeted():
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    model = BennettRPE.from_mb(mb, sparsity=0.05, seed=0)
    # 144 KCs * 5% = 7
    assert model.coder.k_active == 7


def test_cs_plus_acquires():
    _, h = _train()
    labels = np.array(h["label"])
    m = np.array(h["m_hat"])
    m_plus = m[labels == "CS+"]
    # MBON response to CS+ rises substantially
    assert m_plus[-1] > 0.7
    assert m_plus[-1] > m_plus[0] + 0.5


def test_dan_error_vanishes_on_cs_plus():
    _, h = _train()
    labels = np.array(h["label"])
    d = np.array(h["dan"])
    d_plus = d[labels == "CS+"]
    # DAN prediction error should decay from ~0.8 to <0.2
    assert abs(d_plus[0]) > 0.5
    assert abs(d_plus[-1]) < 0.2


def test_cs_minus_does_not_grow():
    _, h = _train()
    labels = np.array(h["label"])
    m = np.array(h["m_hat"])
    m_minus = m[labels == "CS-"]
    # CS- response stays bounded — minor shared-KC drift is expected, no acquisition
    assert m_minus[-1] < 0.3
    assert m_minus[-1] <= m_minus[0] + 0.05
