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

### A connectome substrate that reveals compartment structure (Fig. 1)

We extracted the larval MB subgraph from Winding 2023 by filtering on the cell-type annotations: 144 Kenyon cells (KCs), 48 MBONs (24 cell types in left/right pairs), 14 DAN cells (7 types — DAN-c1/d1/f1/g1 aversive and DAN-i1/j1/k1 appetitive per Saumweber 2018, Weiglein 2024 and Hu 2025), 4 OAN cells, and 206 projection neurons — 588 neurons connected by 22,331 synaptic edges (Fig. 1A). DAN-h1, present in older anatomical descriptions, is absent from the L1 EM volume (Eichler 2017 confirms this developmental absence).

To test whether the wiring alone encodes the canonical compartmental organisation of Eichler 2017, we computed the bipartite Jaccard similarity between each DAN and each MBON on their shared KC pools (Fig. 1B), then thresholded and hierarchically reordered. The block-diagonal structure is immediately visible. The single highest-overlap pair is DAN-i1 ↔ MBON-i1 with J = 0.82 (Fig. 1C); the same pattern repeats across hemispheres. **Anatomical compartment identity is recoverable from synaptic wiring alone.** This validates the substrate for the modelling that follows.

Onto this substrate we built four agents (Fig. 1D), each sharing the same PN→KC sparse coder (top-k=7 of 144 active per odor, matching Honegger 2014's 5% sparsity), and each differing only in the KC→MBON learning rule and the DAN signal. The four are: (i) classical RPE (Bennett 2021 single-valence with signed weights), (ii) a Bayesian-observer "AIF" agent tracking posterior entropy and surprisal, (iii) a precision-weighted hybrid `η_eff = η_0 (1 + λ·surprisal)`, and (iv) a full Bennett-MV dual-valence model with parallel appetitive and aversive traces. All four are tested on identical behavioural protocols.

### Signed-weight RPE supports acquisition and extinction (Fig. 2)

On a standard alternating CS+/CS- conditioning task with reward paired to CS+, the RPE agent acquires the appetitive response cleanly: across 20 random seeds, m_hat for CS+ rises from 0.20 ± 0.02 to 0.94 ± 0.01 over 30 trials, with CS- staying flat near zero (Fig. 2A). The DAN error (r − m_hat) decays from 0.82 to 0.06 (Fig. 2B), confirming the standard Rescorla-Wagner trajectory. When the contingency is reversed (CS+ → no reward), the model extinguishes 0.94 → 0.15 over 20 trials (Fig. 2C) — but only because signed weights are permitted. Clipping weights at zero (a common implementation choice we removed in our rigor pass) makes extinction structurally impossible: w+ can only grow, never shrink. **Signed-weight RPE is necessary for acquisition and extinction.**

### A Bayesian surprisal channel is necessary for novelty signalling (Fig. 3)

After training, we presented familiar and novel-odor probes without reward (no learning during probe phase). The RPE agent's |DAN| on familiar CS+ is large (∼0.5 on average due to extinction-style PE when reward is omitted) but contains no novelty signal — its DAN magnitude is bounded by learned m_hat alone (Fig. 3A). The AIF agent shows the opposite profile: confident posterior (entropy ≈ 0) on familiar odors, but **bimodal** novelty response on 200 unique probes (Fig. 3B) — approximately half cluster at zero (novel KC patterns that happen to align with one trained class) and approximately half cluster at log(2) ≈ 0.69 (the theoretical maximum entropy for two classes, achieved when the KC pattern strongly mixes features of both). Repeated exposure to a single novel odor with learning enabled produces immediate habituation: AIF DAN drops from log(2) on the first trial to 0 on the second (Fig. 3C), reproducing the canonical Hattori 2017 α'3 novelty-habituation profile. **A Bayesian surprisal channel — explicitly tracking posterior entropy or model-evidence surprisal − log p(KC) − is necessary for these signatures.**

### Precision-weighted learning yields latent inhibition (Fig. 4)

Pre-exposure to an odor without reward is known to slow subsequent CS+US learning (latent inhibition, Jacob & Waddell 2022). The mechanistic prediction of any precision-weighted theory is direct: pre-exposure habituates surprisal, lowering η_eff at the moment when reward is later introduced. We tested this with a two-group design: pre-exposed (20 unrewarded CS+ presentations) vs. control (20 unrelated-odor presentations), each followed by 40-trial CS+/CS- acquisition (Fig. 4A,B). Across 20 seeds, RPE shows essentially overlapping acquisition curves between groups — the structural prediction for a model with no precision-weighting mechanism. The Dual-DAN model shows a dramatic separation: the control group reaches m_hat = 1.0 by trial 1, while the pre-exposed group ramps slowly from 0 to ∼0.8 over 40 trials. A λ-sweep (Fig. 4C) confirms the causal mechanism: the LI effect rises monotonically from 0.06 at λ = 0 (the pure-RPE limit) to 0.89 at λ = 5. **Precision-weighted learning rate is necessary for latent inhibition.**

### Parallel valence traces enable fast reversal (Fig. 5)

The Mancini 2019 reversal protocol exposes a fourth phenomenon that single-valence models cannot capture. After 30 trials of CS_A → reward and CS_B → nothing, the contingency reverses (CS_A → nothing, CS_B → reward) and the trial budget is symmetric. The four models diverge sharply (Fig. 5A–C): pure RPE reverses symmetrically (∼10 trials each direction, as expected from a single trace with identical η governing both directions); AIF fails entirely (accumulated Beta-Bernoulli evidence locks the class assignment for 50+ reversal trials); the Dual-DAN hybrid reverses but slowly (the surprisal boost is gone during reversal because both patterns are now familiar); only the **Dual-Valence** Bennett-MV model reverses cleanly via Felsenberg 2018's parallel-opposing-memories mechanism (Fig. 5D). The trace decomposition makes the mechanism explicit: when CS_A flips from rewarded to unrewarded, m+(CS_A) persists at the acquired level — the original appetitive trace is **not erased** — but m-(CS_A) grows in parallel, driven by the cross-valence DAN feedback d- = ReLU(w_M · m+) firing on the now-unrewarded probe. Net m_hat = m+ − m- collapses fast because two things are moving simultaneously instead of one. **Cross-valence DAN feedback with parallel appetitive/aversive traces is necessary for reversal at acquisition speed.** Quantitatively matching Mancini's specific 3:1 acquisition-vs-reversal ratio requires asymmetric appetitive-vs-aversive plasticity parameters (consistent with Davis 2023's observation that aversive memories are more stable), which we leave for future work.

### Necessity is jointly satisfied, not pairwise

Across the four protocols, no subset of three architectural ingredients suffices: signed-weight RPE fails on novelty and latent inhibition; surprisal-channel-only fails on extinction and reversal; precision-weighting alone fails on reversal; parallel valence traces alone fail on latent inhibition. **All four ingredients are individually necessary, and the four together are sufficient for the four signatures tested.** The connectome substrate does not pick the learning rule — behaviour does.

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
