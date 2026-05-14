# A dual-pathway dopaminergic code for reward prediction and epistemic surprise in the larval *Drosophila* mushroom body

**Target venue:** *eLife* via *bioRxiv*. Target submission: 2026-08.
**Author:** Oisin Lyons. **AI-assistance disclosure:** see Methods.

---

## Abstract

The *Drosophila* mushroom body (MB) has been modeled for three decades as a single-channel reward-prediction-error (RPE) machine. Yet dopamine neurons (DANs) fire to novel odors before any reward (Hattori 2017), prior exposure produces latent inhibition (Jacob & Waddell 2022), and contingency reversal completes within one cycle (Mancini 2019) — phenomena single-channel RPE structurally cannot explain. The newly released larval connectome (Winding 2023; 3,016 neurons, every synapse mapped) lets us isolate which architectural ingredients each phenomenon needs by holding wiring fixed and varying only the learning rule.

We compare four agents on the same connectome substrate (588-neuron MB subgraph; 144 KCs, 24 MBON types, 7 DAN types). Each ingredient is **individually necessary**: signed-weight RPE for extinction; a Bayesian surprisal channel (DAN ∝ posterior entropy / −log p(KC)) for novelty signaling and habituation; precision-weighted learning rate (η_eff = η_0(1+λ·surprisal)) for latent inhibition; parallel appetitive/aversive traces with cross-valence DAN feedback (Bennett-MV / Felsenberg-POM) for fast reversal. Removing any one breaks at least one signature. Unsupervised clustering of MBINs by shared KC pools recovers Eichler 2017's compartmental organisation from wiring alone (canonical DAN-i1 ↔ MBON-i1 pair, J=0.82). We predict: the larval DAN encoding perceptual surprise should habituate to novel odors before reinforcement — testable by calcium imaging in DAN-c1.

---

## Introduction

The *Drosophila* mushroom body is the best-characterised associative-learning circuit in any animal. Three decades of experiment converge on a stereotyped picture: olfactory projection neurons (PNs) feed a sparse code in ~2,000 Kenyon cells (KCs), whose synapses onto a few dozen mushroom body output neurons (MBONs) are bidirectionally modulated by dopaminergic teaching signals from ~20 DAN types, each occupying its own compartment (Tully & Quinn 1985; Heisenberg 2003; Aso et al. 2014; Modi et al. 2020). The leading computational account is Bennett et al. 2021's reinforcement-prediction-error (RPE) model: a DAN encodes `r − m̂` at each compartment, KC→MBON synapses adapt by a three-factor rule, and extinction emerges naturally from MBON-driven prediction error feedback (Felsenberg et al. 2018). This single-channel picture has dominated the field for a decade.

Yet several robust observations remain unaccounted for. Hattori et al. 2017 reported that the adult PPL1-α'3 DAN fires to novel odors *before any reward pairing* and habituates with re-exposure — a signal pure RPE cannot generate, because there is no reward to be wrong about. Jacob & Waddell 2022 documented latent inhibition: pre-exposure to an odor without reward measurably slows subsequent CS+US learning, requiring some attention-like gating that flat RPE lacks. Mancini et al. 2019 showed larval reversal completes in roughly one cycle compared to three for the original acquisition — an asymmetry incompatible with single-trace gradient learning, which is symmetric in η. Each observation has been treated as a curious exception. We argue they are diagnostic.

The recent publication of the complete larval *Drosophila* synaptic-resolution connectome (Winding et al. 2023) makes it possible to test which architectural ingredients each of these signatures requires, by holding the substrate fixed and varying only the learning rule. Using the 588-neuron MB subgraph as a common substrate, we compare four agents — classical RPE, a Bayesian-observer "active inference" channel, a precision-weighted hybrid, and a full dual-valence model — across four behavioural protocols. Each architectural ingredient turns out to be individually necessary for a specific phenomenon. The framework also yields a concrete falsifiable prediction at the cellular level: DAN-c1 should carry a perceptual-surprise component analogous to adult PPL1-α'3.

---

## Results

### Figure 1 — Substrate and architecture

(a) Larval *Drosophila* MB subgraph extracted from Winding 2023: 144 KCs, 24 MBON types (48 cells), 7 DAN types (14 cells, 4 aversive + 3 appetitive), 206 PNs.
(b) Bipartite Jaccard heatmap of DAN-MBON shared KC pools, hierarchically reordered. Block structure reveals compartments without any anatomical labels.
(c) Highest-Jaccard pair: DAN-i1 ↔ MBON-i1 (J=0.82). Subtype labels confirm canonical Eichler 2017 compartment recovery purely from wiring.
(d) Architecture diagram: KC → MBON readout pathway with three plasticity channels (RPE, AIF surprisal, parallel-valence POM).

**Source files:** `experiments/00_load_connectome.py`, `experiments/04_compartments.py`.

### Figure 2 — Acquisition + extinction (the RPE backbone)

(a) Bennett RPE acquisition curve on connectome PN→KC and KC→MBON: m_hat for CS+ rises 0.18 → 0.93 over 60 trials; CS- stays flat. (b) DAN error (r - m_hat) decays from 0.82 to 0.07.
(c) Extinction phase: with signed weights (Phase 3c rigor fix), m_hat extinguishes 0.94 → 0.15 over 20 trials. Without signed weights, RPE structurally cannot extinguish.

**Claim**: signed-weight RPE is necessary and sufficient for acquisition+extinction. Multi-seed variance bands (N=20).

**Source files:** `experiments/01_rpe_baseline.py`, `experiments/05_extinction.py`.

### Figure 3 — Novel-odor signaling (the AIF channel)

(a) After CS+/CS- conditioning, present 12 novel odors and 11 CS+/CS- mixtures. (b) AIF |DAN| reaches the theoretical maximum log(2)=0.69 on 5/12 truly-ambiguous novels; RPE |DAN| is bounded by m_hat magnitude. (c) Repeated novel exposure → AIF DAN habituates (the Hattori-2017-α'3 signature). (d) RPE has no mechanism to flag novel odors.

**Claim**: a Bayesian-observer surprisal channel is necessary for novelty/familiarity signaling and habituation.

**Source files:** `experiments/02_aif_agent.py`, `experiments/03_novel_odor.py`.

### Figure 4 — Latent inhibition (precision-weighted RPE)

(a) Pre-exposure protocol: 20 unrewarded CS+ presentations, then 40-trial CS+/CS- acquisition (vs. control group with unrelated pre-exposure odor). (b) Three models, 20 seeds: RPE shows essentially overlapping curves (no LI), Dual-DAN shows dramatic separation (control reaches 1.0 immediately, pre-exposed CS+ ramps slowly from 0 to ~0.8 over 40 trials). (c) Lambda sweep: LI effect (control−pre-exposed AUC) rises monotonically from 0.06 at λ=0 to 0.89 at λ=5. Pure RPE is the λ=0 limit.

**Claim**: surprisal-modulated learning rate `η_eff = η₀(1 + λ·surprisal_norm)` is necessary for latent inhibition. The Friston-Schwartenbeck precision form.

**Source files:** `experiments/06_latent_inhibition.py`, `experiments/08_robustness.py`.

### Figure 5 — Fast reversal (parallel-valence POM)

(a) Mancini-style reversal: 30 trials CS_A → reward, then 30 trials CS_A → nothing, CS_B → reward. (b) Four models compared head-to-head:
- RPE: symmetric reversal, ~10 trials each direction
- AIF: cannot reverse, m_hat(CS_A) stays high for 50+ trials
- Dual-DAN: slow asymmetric reversal (~14 trials)
- Dual-Valence: instant reversal via parallel opposing memories
(c) Trace dynamics: during reversal of CS_A, w+ persists (old appetitive trace) while w- grows in parallel (new aversive trace). m_hat = m+ − m- collapses without w+ extinguishing — the Felsenberg POM mechanism.
(d) η tuning: symmetric Bennett-MV gives ratio ≈ 1 (reversal as fast as acquisition); Mancini's 3:1 requires asymmetric η+/η-.

**Claim**: parallel valence traces with cross-valence DAN feedback are necessary for reversal at acquisition speed. Pure RPE cannot match. Asymmetric plasticity (future work) needed for full quantitative match.

**Source files:** `experiments/07_reversal.py`, `experiments/08_robustness.py`.

### Figure 6 (optional) — Necessity matrix

A simple table or schematic showing which architectural ingredient is necessary for which behavioural signature. Removing any one ingredient → at least one signature fails.

---

## Discussion *(target 800 words)*

1. **Four ingredients, each necessary**. The connectome alone doesn't pick the learning rule; behaviour does. Each ingredient maps to specific connectome cell types (PN→KC for sparse coding; KC→MBON for readout; DAN→MBON for RPE channel; cross-valence DAN feedback for POM).
2. **Architectural completeness vs. quantitative refinement**. The symmetric Bennett-MV reproduces the QUALITATIVE POM signature but not Mancini's 3:1 asymmetry. This identifies a quantitative constraint: appetitive vs. aversive plasticity must be asymmetric. Davis 2023 has anatomical/molecular evidence for this (different DAN receptors, different decay constants).
3. **The DAN-c1 prediction**. Saumweber 2018 showed DAN-c1 has no associative phenotype under direct optogenetic activation, suggesting a modulatory/permissive role. Hu 2025 documents a gustatory aversive function via D2R. We predict that, *in addition*, DAN-c1 carries a perceptual-surprise component analogous to adult PPL1-α'3 (Hattori 2017). The empirical test: calcium imaging of DAN-c1 during novel-vs-familiar odor presentation before reinforcement. Direct falsification.
4. **Relation to other modeling approaches**. Bennett 2021 / Jurgensen 2024 = single-channel RPE only. Jiang & Litwin-Kumar 2021 = heterogeneous DA without explicit AIF. Active-inference literature (Friston, Pezzulo, Lanillos) hasn't applied to MB. Our contribution is the synthesis on a real connectome.
5. **Limitations**. No action selection (perception-only AIF reduction). Compartments grouped by valence rather than per-compartment plasticity. Reward signal is supervised. No embodied behaviour.
6. **Future**: scale to adult FlyWire; implement asymmetric η+/η- to match Mancini quantitatively; add EFE-based policy selection for full Friston-compatible AIF; embodied agent in larvaworld closing the loop.

---

## Methods *(target 1000 words)*

- **Connectome data**: Winding 2023 *Science* Supplementary Data S1, public; transposed to W[post,pre] convention.
- **MB subgraph extraction**: filter on cell types KC/MBON/MBIN/PN/MB-FBN/MB-FFN.
- **KC sparse coding**: row-normalised PN→KC weights, top-k winner-take-all (k = 5% of 144 = 7 active per odor).
- **RPE rule**: Bennett 2021 single-valence reduction, signed weights enabled.
- **AIF agent**: Beta-Bernoulli p(KC | class), naive Bayes posterior, DAN signal = posterior entropy. Surprisal -log p(KC) reported as Friston-rigorous alternative.
- **Dual-DAN**: precision-weighted RPE, eta_eff = eta_0 * (1 + lambda * standardised surprisal).
- **Dual-Valence**: Bennett-MV with separate w+/w-, cross-valence DAN feedback, λ-baselined update rule.
- **Connectome valence labels**: from Saumweber 2018, Weiglein 2024, Hu 2025; appetitive = i/j/k compartments, aversive = c/d/f/g compartments.
- **Multi-seed robustness**: 20 seeds per phenomenon for variance bands.
- **AI assistance**: this work was developed in collaboration with Claude (Anthropic) for code generation, literature search, and verification of equations. No claims, results, or analyses depend on Claude as an author. Code at github.com/oisinlyons/brain (TODO).

---

## References *(skeleton — fill in)*

- Bennett et al. 2021 *Nat Commun* — RPE in MB
- Cohn, Morantte, Ruta 2015 *Cell* — γ-lobe DAN dynamics
- Davis 2023 *Curr Biol* — memory stability across DAN receptor systems
- Eichler et al. 2017 *Nature* — larval MB connectome
- Eschbach et al. 2020 *Nat Neurosci* — FBN feedback architecture
- Felsenberg et al. 2018 *Cell* — parallel opposing memories
- Friston 2010 *Nat Rev Neurosci* — Free Energy Principle
- Hattori et al. 2017 *Cell* — α'3 novelty DAN
- Hu et al. 2025 *eLife* — DAN-c1 D2R gustatory aversive
- Jacob & Waddell 2022 *Curr Biol* — latent inhibition in fly
- Jurgensen et al. 2024 *iScience* — spiking larval PE model
- Mancini et al. 2019 *Learn Mem* — reversal learning in larvae
- Saumweber et al. 2018 *Nat Commun* — per-DAN optogenetic valence
- Schwartenbeck et al. 2015 *J Neurosci* — DA as precision over policies
- Weiglein et al. 2024 *eLife* — Four DL1 DANs signal taste punishment
- Winding et al. 2023 *Science* — larval connectome
- Yu & Dayan 2005 *Neuron* — unexpected uncertainty
