"""Phase 2 regression tests for the AIF agent."""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair


def _train(n_trials: int = 60, seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(aif.coder.n_pn, active_frac=0.1, seed=seed + 1)
    h = {"label": [], "m_hat": [], "dan": []}
    for odor, reward, label in conditioning_trials(cs, n_trials):
        out = aif.step(odor, reward)
        h["label"].append(label); h["m_hat"].append(out["m_hat"]); h["dan"].append(out["dan"])
    return aif, h


def test_aif_acquires_cs_plus():
    _, h = _train()
    labels = np.array(h["label"])
    m_plus = np.array(h["m_hat"])[labels == "CS+"]
    assert m_plus[-1] > 0.95   # AIF should converge to ~1.0 on CS+


def test_aif_cs_minus_low():
    _, h = _train()
    labels = np.array(h["label"])
    m_minus = np.array(h["m_hat"])[labels == "CS-"]
    assert m_minus[-1] < 0.1


def test_aif_dan_starts_uncertain():
    """First trial has no data — posterior is the smoothed prior, entropy ~ log(2)."""
    _, h = _train(n_trials=2)
    d = np.array(h["dan"])
    # ~0.69 for two classes with add-one smoothing
    assert 0.4 < d[0] < 0.75


def test_aif_dan_on_familiar_falls_to_zero():
    _, h = _train()
    d_end = np.array(h["dan"])[-6:]   # last 6 trials
    assert d_end.max() < 0.05
