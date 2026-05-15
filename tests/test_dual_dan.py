"""Phase 3d regression tests: dual-DAN model and latent inhibition.

These tests pin down the headline behaviour:
  - Dual model acquires CS+ on a control (un-pre-exposed) schedule
  - Dual model shows latent inhibition: pre-exposed CS+ acquires slower
  - Pure RPE does NOT show latent inhibition (the structural prediction)
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.dual_dan import DualDANAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import make_cs_pair
from mb_dual_dan.tasks.latent_inhibition import make_li_schedule


def _setup(seed: int = 0):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    n_pn = BennettRPE.from_mb(mb, seed=seed).coder.n_pn
    cs = make_cs_pair(n_pn, active_frac=0.1, seed=seed + 1)
    return mb, n_pn, cs


def _li_curves(model_fn, mb, n_pn, cs, seed):
    """Run the model on both LI groups, return (control_slope, preexposed_slope) on CS+ acq."""
    out = {}
    for preexpose, group_name in [(True, "pre"), (False, "ctrl")]:
        agent = model_fn()
        sched = make_li_schedule(cs, n_pn, n_preexposure=20, n_acquisition=40,
                                 preexpose_cs_plus=preexpose, seed=seed + 11)
        for odor, reward, label in sched.pre_exposure:
            agent.step(odor, reward)
        cs_plus_log = []
        for odor, reward, label in sched.acquisition:
            o = agent.step(odor, reward)
            if label == "CS+":
                cs_plus_log.append(o["m_hat"])
        out[group_name] = np.array(cs_plus_log)
    # slope = m_hat at trial 5 minus trial 0 (early acquisition rate)
    return out["ctrl"][5] - out["ctrl"][0], out["pre"][5] - out["pre"][0]


def test_dual_acquires_cs_plus_under_control():
    """Without pre-exposure, the dual model should acquire CS+ fast."""
    mb, n_pn, cs = _setup()
    ctrl, _ = _li_curves(
        lambda: DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=1.5, sparsity=0.05, seed=0),
        mb, n_pn, cs, seed=0)
    assert ctrl > 0.5


def test_dual_shows_latent_inhibition():
    """Pre-exposing CS+ before reward pairing should slow acquisition by ~3x."""
    mb, n_pn, cs = _setup()
    ctrl, pre = _li_curves(
        lambda: DualDANAgent.from_mb(mb, eta0=0.025, lambda_precision=1.5, sparsity=0.05, seed=0),
        mb, n_pn, cs, seed=0)
    li_effect = ctrl - pre
    assert li_effect > 0.3, f"Dual model should show clear LI; got effect {li_effect:.3f}"


def test_pure_rpe_shows_no_latent_inhibition():
    """The structural prediction: pure RPE has no mechanism for LI.
    Control and pre-exposed slopes should be within noise of each other."""
    mb, n_pn, cs = _setup()
    ctrl, pre = _li_curves(
        lambda: BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=0),
        mb, n_pn, cs, seed=0)
    li_effect = ctrl - pre
    assert abs(li_effect) < 0.15, f"Pure RPE should show ~0 LI; got {li_effect:.3f}"


def test_dual_dan_signal_carries_both_channels():
    """A single step() should expose both RPE and AIF DAN signals plus surprisal."""
    mb, _, cs = _setup()
    agent = DualDANAgent.from_mb(mb, sparsity=0.05, seed=0)
    out = agent.step(cs.cs_plus, reward=1.0)
    for key in ("dan_rpe", "dan_aif", "surprisal", "surprisal_norm", "eta_effective"):
        assert key in out
    # First trial: surprisal is high; eta_effective should be > eta0
    assert out["eta_effective"] >= agent.eta0
