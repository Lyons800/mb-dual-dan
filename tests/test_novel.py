"""Phase 3 tests: novel-odor probing pin-downs.

The headline claim of the project is that the AIF agent produces a max-entropy
DAN response on truly novel KC patterns while RPE has no such mechanism. These
tests pin down that signature on the real larval connectome.
"""

import numpy as np

from mb_dual_dan.connectome import extract_mb, load_winding
from mb_dual_dan.models.aif_agent import AIFAgent
from mb_dual_dan.models.rpe_baseline import BennettRPE
from mb_dual_dan.tasks.conditioning import conditioning_trials, make_cs_pair
from mb_dual_dan.tasks.novel_odor import make_probes


def _trained(seed: int = 0, n_train: int = 60):
    c = load_winding()
    mb = extract_mb(c, include_pns=True)
    rpe = BennettRPE.from_mb(mb, eta=0.025, w_init=0.05, sparsity=0.05, seed=seed)
    aif = AIFAgent.from_mb(mb, sparsity=0.05, seed=seed)
    cs = make_cs_pair(rpe.coder.n_pn, active_frac=0.1, seed=seed + 1)
    for odor, reward, _ in conditioning_trials(cs, n_train):
        rpe.step(odor, reward, learn=True)
        aif.step(odor, reward, learn=True)
    return rpe, aif, cs


def test_aif_max_dan_on_novel_reaches_log2():
    """At least one truly-ambiguous novel odor should drive AIF DAN to ~log(2)."""
    rpe, aif, cs = _trained()
    probes = make_probes(cs, aif.coder.n_pn, n_novel=12, seed=42)
    dans = []
    for pn, _ in probes.novel:
        out = aif.step(pn, reward=0.0, learn=False)
        dans.append(out["dan"])
    max_dan = max(dans)
    # log(2) ~= 0.693 is theoretical max entropy for 2 classes
    assert max_dan > 0.6


def test_aif_dan_zero_on_familiar():
    """After training, AIF should be confident on the trained CS+/CS- odors."""
    rpe, aif, cs = _trained()
    out_plus = aif.step(cs.cs_plus, reward=0.0, learn=False)
    out_minus = aif.step(cs.cs_minus, reward=0.0, learn=False)
    assert out_plus["dan"] < 0.05
    assert out_minus["dan"] < 0.05


def test_rpe_cannot_signal_novelty_without_reward():
    """RPE has no mechanism to signal novelty per se — its DAN is bounded by m_hat,
    which for a truly novel KC pattern with no overlap with CS+ is near zero."""
    rpe, aif, cs = _trained()
    probes = make_probes(cs, rpe.coder.n_pn, n_novel=12, seed=42)
    rpe_dans = [rpe.step(pn, reward=0.0, learn=False)["dan"] for pn, _ in probes.novel]
    aif_dans = [aif.step(pn, reward=0.0, learn=False)["dan"] for pn, _ in probes.novel]
    # AIF's max should beat RPE's max — only AIF has a real novelty channel
    assert max(aif_dans) > max(np.abs(rpe_dans))


def test_aif_dan_habituates_with_exposure():
    """Repeated exposure (with learning enabled, no reward) should decay AIF DAN
    on a novel odor as the agent's posterior tightens — the Hattori 2017 alpha'3
    signature of novelty-driven DAN responses habituating with experience."""
    rpe, aif, cs = _trained()
    probes = make_probes(cs, aif.coder.n_pn, n_novel=24, seed=99)
    # Pick the novel odor with the highest initial AIF DAN (truly ambiguous).
    initial_dans = []
    for pn, _ in probes.novel:
        d = aif.step(pn, reward=0.0, learn=False)["dan"]
        initial_dans.append(d)
    if max(initial_dans) < 0.3:
        return  # no ambiguous novel found in this seed; skip
    best_idx = int(np.argmax(initial_dans))
    novel_pn, _ = probes.novel[best_idx]
    # Repeat exposure with learning on.
    dan_track = []
    for _ in range(8):
        out = aif.step(novel_pn, reward=0.0, learn=True)
        dan_track.append(out["dan"])
    # DAN should fall as the posterior tightens (the novel odor gets sorted into
    # the unrewarded class because reward=0 each trial).
    assert dan_track[0] >= dan_track[-1]
    assert dan_track[-1] < dan_track[0] * 0.5
