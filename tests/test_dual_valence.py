"""Phase 3i regression tests: dual-valence Bennett-MV / Felsenberg-POM model.

Headline behaviour: parallel appetitive (w+) and aversive (w-) traces, with
cross-valence DAN feedback (M+ excites D-, M- excites D+). This is the
mathematical instantiation of Felsenberg 2018's parallel-opposing-memories.

The diagnostic for "POM working": during reversal, the OLD w+ trace
persists (m+ stays high for a few trials) while a NEW w- trace forms in
parallel. Behaviour (m_hat = m+ - m-) reverses fast even though w+ has not
extinguished — the asymmetry Mancini 2019 measured.
"""

import numpy as np

from brain.connectome import extract_mb, load_winding
from brain.models.dual_valence import DualValenceMB
from brain.tasks.conditioning import conditioning_trials, make_cs_pair, reversal_trials


def _setup(seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    agent = DualValenceMB.from_mb(mb, eta=0.025, lambda_baseline=1.0, w_M=1.0,
                                  w_init=0.05, sparsity=0.05, seed=seed)
    cs = make_cs_pair(agent.coder.n_pn, active_frac=0.1, seed=seed + 1)
    return agent, cs


def test_valence_partition_matches_eichler():
    """Connectome-derived MBON valence partition should split into known
    appetitive (compartments i, j, k) and aversive (c, d, f, g) populations."""
    agent, _ = _setup()
    s = agent.valence_summary()
    assert s["n_mbon_appetitive"] >= 4    # MBONs in i, j, k compartments
    assert s["n_mbon_aversive"] >= 8      # MBONs in c, d, f, g compartments
    # rest are calyx + e + m + untyped
    assert s["n_mbon_neutral"] >= 10


def test_dual_valence_acquires_appetitive_trace():
    """After CS+ paired with reward, m+ on CS+ should rise — and m_hat too."""
    agent, cs = _setup()
    for odor, reward, _ in conditioning_trials(cs, 30):
        agent.step(odor, reward)
    out = agent.step(cs.cs_plus, 0.0, learn=False)
    assert out["m_plus"] > 0.5
    assert out["m_hat"] > 0.0


def test_dual_valence_reverses_via_parallel_aversive_trace():
    """The Felsenberg POM signature: after reversal of CS_A (reward removed,
    reward shifted to CS_B), m_hat(CS_A) flips sign — driven by the NEW w-
    trace forming, not by w+ extinguishing."""
    agent, cs = _setup()
    # acquisition
    for odor, reward, _ in conditioning_trials(cs, 30):
        agent.step(odor, reward)
    # snapshot CS_A response before reversal
    snap_before = agent.step(cs.cs_plus, 0.0, learn=False)
    # reversal phase
    for odor, reward, _ in reversal_trials(cs, 20):
        agent.step(odor, reward)
    snap_after = agent.step(cs.cs_plus, 0.0, learn=False)
    # m_hat for CS_A should drop substantially after reversal
    assert snap_before["m_hat"] > 0.3
    assert snap_after["m_hat"] < snap_before["m_hat"] - 0.3
    # The KEY signature: m_minus should have GROWN (new aversive trace
    # parallel to the original appetitive one).
    assert snap_after["m_minus"] > snap_before["m_minus"]


def test_dual_valence_reverses_faster_than_pure_rpe():
    """The discriminating prediction from Bennett-MV / Felsenberg-POM:
    dual-valence reversal completes in fewer trials than pure RPE."""
    from brain.models.rpe_baseline import BennettRPE

    c = load_winding()
    mb = extract_mb(c, include_pns=True)

    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=0)
    dv = DualValenceMB.from_mb(mb, eta=0.025, sparsity=0.05, seed=0)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=1)

    for odor, reward, _ in conditioning_trials(cs, 30):
        rpe.step(odor, reward); dv.step(odor, reward)

    def trials_to_cross(agent, cs, threshold=0.3):
        for t, (odor, reward, _) in enumerate(reversal_trials(cs, 30)):
            agent.step(odor, reward)
            m = agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"]
            if m < threshold:
                return t
        return 30

    t_rpe = trials_to_cross(rpe, cs)
    t_dv = trials_to_cross(dv, cs)
    assert t_dv <= t_rpe, f"DualValence should reverse no slower than RPE; got DV={t_dv}, RPE={t_rpe}"


def test_cross_valence_dan_feedback():
    """M+ should excite D- and vice versa (Bennett Eq. 13). Verify by
    presenting a pattern that activates trained appetitive MBONs and
    checking that d_minus rises proportionally with m_plus."""
    agent, cs = _setup()
    # train so w+ grows for CS_A
    for odor, reward, _ in conditioning_trials(cs, 30):
        agent.step(odor, reward)
    # probe with reward = 0 — d_minus should still be > 0 from m_plus feedback
    out = agent.step(cs.cs_plus, reward=0.0, learn=False)
    assert out["m_plus"] > 0.0
    assert out["d_minus"] > 0.0
    # cross-valence: d_minus = ReLU(r- + w_M * m+) and r- = 0, w_M = 1 => d_minus = m+
    assert abs(out["d_minus"] - out["m_plus"]) < 0.01
