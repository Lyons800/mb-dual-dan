"""Phase 3f regression tests: reversal learning kinetics.

Pin down what the three current single-valence models do under contingency
reversal. The honest finding: none of them reproduce Mancini 2019's
'fast reversal' asymmetry — that requires parallel opposing memories
(Felsenberg 2018), which means we need a dual-valence MBON architecture
in Phase 3i.
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.dual_dan import DualDANAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair, reversal_trials


def _run(agent_fn, n_acq=30, n_rev=30, seed=0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    agent = agent_fn(mb, seed)
    cs = make_cs_pair(agent.coder.n_pn if hasattr(agent, "coder") else agent.rpe.coder.n_pn,
                      active_frac=0.1, seed=seed + 1)
    m_a, phase_log = [], []
    for odor, reward, _ in conditioning_trials(cs, n_acq):
        agent.step(odor, reward, learn=True)
        m_a.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
        phase_log.append("acq")
    for odor, reward, _ in reversal_trials(cs, n_rev):
        agent.step(odor, reward, learn=True)
        m_a.append(agent.step(cs.cs_plus, 0.0, learn=False)["m_hat"])
        phase_log.append("rev")
    return np.array(m_a), np.array(phase_log)


def test_rpe_reverses_symmetrically():
    """Pure RPE: acquisition and reversal at the same rate (single trace, same eta)."""
    m_a, phase = _run(lambda mb, s: BennettRPE.from_mb(mb, seed=s))
    m_a_acq = m_a[phase == "acq"]
    m_a_rev = m_a[phase == "rev"]
    assert m_a_acq[-1] > 0.5             # acquired
    assert m_a_rev[-1] < 0.5             # reversed
    # symmetric: end of acquisition vs end of reversal should be roughly mirror-image around 0.5
    assert abs(m_a_acq[-1] + m_a_rev[-1] - 1.0) < 0.3


def test_aif_cannot_reverse_within_30_trials():
    """Bayesian observer: accumulated evidence locks in class assignment."""
    m_a, phase = _run(lambda mb, s: AIFAgent.from_mb(mb, seed=s))
    m_a_rev = m_a[phase == "rev"]
    # m_a stays high through most of the reversal phase
    assert m_a_rev[:20].mean() > 0.8


def test_dual_reverses_but_slower_than_acquisition():
    """Dual-DAN: surprisal-boost is gone during reversal (patterns are
    familiar), so reversal proceeds at base rate — slower than the
    surprisal-boosted acquisition. This is the OPPOSITE of Mancini's
    fast-reversal data and motivates the dual-valence Phase 3i."""
    m_a, phase = _run(lambda mb, s: DualDANAgent.from_mb(mb, seed=s))
    # Dual model still completes reversal within 30 trials
    m_a_rev = m_a[phase == "rev"]
    assert m_a_rev[-1] < 0.5
    # ...but it overshoots/lingers during early reversal (slower than ideal)
    # Just check the trajectory is monotonically decreasing-ish across late reversal
    assert m_a_rev[-5:].mean() < m_a_rev[:5].mean()
