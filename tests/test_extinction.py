"""Phase 3c regression tests: extinction behaviour.

Documents the complementary failure modes of RPE and the Bayesian-observer
AIF formulation:
- RPE extinguishes correctly under signed weights (allow_negative_weights=True).
- AIF (perceptual-Bayesian) cannot extinguish; this is informative, not a bug.
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, extinction_trials, make_cs_pair


def _setup(seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)
    return rpe, aif, cs


def _run(rpe, aif, cs, n_acq, n_ext):
    plus_m_rpe, plus_m_aif = [], []
    plus_phase = []
    for odor, reward, label in conditioning_trials(cs, n_acq):
        rs = rpe.step(odor, reward); a = aif.step(odor, reward)
        if label.startswith("CS+"):
            plus_m_rpe.append(rs["m_hat"]); plus_m_aif.append(a["m_hat"]); plus_phase.append("acq")
    for odor, reward, label in extinction_trials(cs, n_ext):
        rs = rpe.step(odor, reward); a = aif.step(odor, reward)
        if label.startswith("CS+"):
            plus_m_rpe.append(rs["m_hat"]); plus_m_aif.append(a["m_hat"]); plus_phase.append("ext")
    return np.array(plus_m_rpe), np.array(plus_m_aif), np.array(plus_phase)


def test_rpe_extinguishes_with_signed_weights():
    """RPE must be able to extinguish — this is the Bennett-faithful behaviour."""
    rpe, aif, cs = _setup()
    m_rpe, _, phase = _run(rpe, aif, cs, n_acq=60, n_ext=40)
    acq_end = m_rpe[phase == "acq"][-1]
    ext_end = m_rpe[phase == "ext"][-1]
    assert acq_end > 0.7, "RPE should acquire CS+"
    assert ext_end < acq_end * 0.4, f"RPE should extinguish substantially: acq_end={acq_end:.3f}, ext_end={ext_end:.3f}"


def test_aif_cannot_extinguish_documented():
    """AIF perceptual model cannot extinguish — documented honestly so future
    fixes (forgetting prior, EFE-driven model revision) have a clear target."""
    rpe, aif, cs = _setup()
    _, m_aif, phase = _run(rpe, aif, cs, n_acq=60, n_ext=40)
    acq_end = m_aif[phase == "acq"][-1]
    ext_end = m_aif[phase == "ext"][-1]
    # AIF stays confident through extinction — m_hat barely budges
    assert acq_end > 0.9
    assert abs(ext_end - acq_end) < 0.1, "Bayesian observer with positive pseudocounts cannot extinguish"


def test_rpe_dan_spikes_negative_at_extinction_onset():
    """When the contingency reverses, RPE DAN must show a large negative PE."""
    rpe, aif, cs = _setup()
    dan_log = []
    phase_log = []
    for odor, reward, label in conditioning_trials(cs, 60):
        d = rpe.step(odor, reward)["dan"]
        if label.startswith("CS+"):
            dan_log.append(d); phase_log.append("acq")
    for odor, reward, label in extinction_trials(cs, 10):
        d = rpe.step(odor, reward)["dan"]
        if label.startswith("CS+"):
            dan_log.append(d); phase_log.append("ext")
    dan_log = np.array(dan_log); phase_log = np.array(phase_log)
    last_acq = dan_log[phase_log == "acq"][-1]
    first_ext = dan_log[phase_log == "ext"][0]
    # last acquisition trial: small positive DAN; first extinction trial: large negative DAN
    assert abs(last_acq) < 0.2
    assert first_ext < -0.5
