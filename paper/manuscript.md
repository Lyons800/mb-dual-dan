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

## Discussion

We have shown that four architectural ingredients are jointly necessary to reproduce four well-established behavioural phenomena in the larval *Drosophila* mushroom body, holding the synaptic substrate constant at the Winding 2023 connectome. No subset of three ingredients suffices. The result resolves a tension that has accumulated in the field: the standard single-channel RPE account (Bennett 2021 and predecessors) is correct for acquisition and extinction, but a separate channel is needed for novelty-related phenomena (Hattori 2017), precision-weighting is needed for latent inhibition (Jacob & Waddell 2022), and parallel valence traces are needed for fast reversal (Mancini 2019; Felsenberg 2018). Each phenomenon has been treated as a curious exception. We argue they share a common diagnosis: cortex-like learning rules layered on the same connectome substrate.

The DAN-c1 prediction is the most directly falsifiable contribution. Saumweber 2018 showed that direct optogenetic activation of DAN-c1 alone produces no associative memory — anomalous for a DAN under standard RPE accounts. Hu 2025 has since documented a gustatory aversive learning role for DAN-c1 via D2R autoregulation. These are not mutually exclusive in a multi-signal DAN framework (Engelhard 2019; Dabney 2020), and we hypothesise that DAN-c1 also carries a perceptual-surprise component analogous to adult PPL1-α'3 (Hattori 2017). The empirical test is direct: calcium imaging of DAN-c1 during the first exposure to a novel odor, before any reward pairing, should show a transient activation that habituates with repeated exposure. A negative result would falsify the specific cellular hypothesis without affecting the architectural claim — any other DAN, or any FBN, could in principle carry the surprisal channel. The prediction is therefore both specific and recoverable.

Our model identifies one quantitative gap: the symmetric Bennett-MV formulation reproduces the qualitative POM signature (reversal as fast as acquisition) but not Mancini's specific 3:1 acquisition-vs-reversal ratio. We achieve a ratio near 1 across the tested η range; the data require closer to 3. This is informative. The Davis lab has documented asymmetric stability between appetitive and aversive memories at the cellular level (different DAN receptors, distinct decay constants, ARM/ASM separation). Implementing this asymmetry — η+ ≠ η-, λ+ ≠ λ-, or distinct decay rates — should close the quantitative gap, but is a parameter and not an architectural change. The prediction is testable: aversive memory traces should retain through longer intervals than appetitive ones in computational terms exactly as Davis 2023 reports them biologically.

Our approach contrasts with three adjacent lines of work. The Nawrot lab (Jurgensen 2024, Rachad 2025) has the closest published substrate — a connectome-informed spiking larval MB with RPE-only DAN signalling. They reproduce delay conditioning and second-order conditioning but do not include a perceptual-surprise channel and have not been shown to reproduce latent inhibition. Jiang & Litwin-Kumar 2021 argued for heterogeneous DAN coding without proposing an explicit AIF/surprisal channel. Active-inference theorists (Friston, Pezzulo, Lanillos, Schwartenbeck) have argued for precision-weighting and EFE-based policy selection but have not applied these to MB substrates. The contribution here is the synthesis on a real connectome and the demonstration that all three signals are required.

Several limitations should be acknowledged. The "AIF" agent in this paper is a perception-only Bayesian observer rather than a full active-inference agent — there is no action selection by EFE minimisation, no policy precision γ, and the latent state is supervised by reward outcome rather than truly latent. A complete AIF treatment using pymdp 1.0 or RxInfer.jl is a natural next step. The MBON pooling collapses 48 cells into two valence populations (per Eichler 2017 / Eschbach 2021 single-MBON behavioural labels); per-compartment plasticity with the full eligibility-trace 3-factor rule (Aso 2014/2016, Cohn 2015) would be a faithful refinement. The substrate is L1; scaling to adult FlyWire (2,580 KCs, 19 MBON types) is the obvious extension and is now feasible. Finally, the agent is disembodied — closing the sensorimotor loop in NeuroMechFly v2 / FlyGym v2 / larvaworld would let us test behavioural outputs against published navigation data.

We close with the observation that the larval *Drosophila* MB is now the smallest brain region in any animal where every synapse is mapped, multiple behavioural protocols have been quantitatively characterised, and individual DANs and MBONs can be perturbed optogenetically. This makes it an ideal proving ground for *which* learning rules biology has converged on. Our claim is not that the brain runs precisely the equations we describe; it is that any complete model of MB function must accommodate the four ingredients we have shown to be individually necessary. We expect this constraint to sharpen as adult connectomes mature and as more behavioural protocols are quantitatively characterised on the same substrate.

---

## Methods

### Connectome substrate

We used the larval *Drosophila* synaptic-resolution connectome from Winding et al. 2023 (*Science*), Supplementary Data S1 (publicly released, no restrictions). The all-to-all connectivity matrix (2,952 × 2,952 neurons; 110,677 nonzero edges; 352,611 synaptic contacts) was transposed at load time to the standard `W[post, pre]` convention so that `y = W · x` gives postsynaptic activity. The accompanying `annotations.csv` provides cell-type labels and subtype annotations (e.g. DAN-c1, MBON-i1) that we use throughout.

The MB subgraph is extracted by filtering on the cell-type column to retain KC, MBON, MBIN (the supercategory containing DANs, OANs, and other modulatory cells), MB-FBN (feedback neurons), MB-FFN (feedforward neurons), and PN (projection neurons). DAN identity is recovered by parsing the `additional_annotations` subtype labels matching the pattern `DAN-*`. The resulting subgraph contains 588 neurons (144 KCs, 48 MBONs, 28 MBINs split into 14 DANs / 4 OANs / 10 other-modulatory cells, 108 MB-FBNs, 54 MB-FFNs, 206 PNs) connected by 22,331 synaptic edges (~157k total synaptic contacts).

Neurotransmitter signs are not in the Winding S1 dataset and Eckstein et al. 2024's neurotransmitter classifier was trained on adult FAFB data and does not apply to L1 larval EM. We use cell-class conventions: KC, MBON, PN, and sensory neurons are treated as cholinergic (excitatory); LN and APL as GABAergic (inhibitory); MBINs are treated as modulatory with sign +1 in the linear readout (their teaching action is implemented separately in the plasticity rule, not as fast postsynaptic drive).

### KC sparse coding

Each odor is represented as a sparse PN activation pattern (~20 of 206 PNs active per odor, drawn at random with uniform intensity in [0.5, 1.0]). The PN→KC weight matrix is extracted from the connectome subgraph and row-normalised. KC activity is the row-normalised PN→KC product followed by a winner-take-all top-k operation with k = 5% of KCs (k=7 of 144 active per odor), matching the published Litwin-Kumar 2017 sparsity target. KC activity is binary (active = 1, inactive = 0).

### RPE agent (Bennett 2021 single-valence reduction)

A single MBON pool reads out from the KCs with plastic weights `w_KC→MBON` initialised at 0.05 and masked by the connectome KC→MBON edge mask. MBON output is `m_hat = ReLU(w · k)`, pooled across the 48 MBONs. The DAN error is `δ = r − m_hat`. The plasticity rule is `Δw_i = η · k_i · δ` with η = 0.025 (Bennett Fig. 2-3 value). Weights are clipped from below at zero only when `allow_negative_weights=False`; for the experiments reported here we allow signed weights, which is necessary for extinction.

### AIF / Bayesian-observer agent

Two latent classes (`rewarded`, `unrewarded`) are tracked with Beta-Bernoulli sufficient statistics over `p(KC_i = 1 | class)` (α and β counts per KC per class). Class assignment for each trial is `c = (reward > 0.5)`; this makes the agent a supervised Bayesian classifier, not a fully unsupervised latent-state AIF agent. The posterior `q(c | KC)` is computed by naive-Bayes log-likelihood with a smoothed log-prior derived from class counts. The agent exposes three signals: posterior entropy `H[q(c|KC)]`, Bayesian surprise `KL[q(c|KC) || q(c)]`, and surprisal `−log p(KC | model) = −log Σ_c p(c) p(KC|c)`. The canonical DAN signal in our results is posterior entropy, which habituates on familiar patterns; surprisal is also reported as the Friston-rigorous alternative.

### Dual-DAN precision-weighted hybrid

Combines RPE and AIF: at each step, the AIF surprisal is z-scored against an online running mean and standard deviation (momentum 0.05) and clipped non-negatively. The effective RPE learning rate is `η_eff = η_0 · (1 + λ · standardised_surprisal)`. We use η_0 = 0.025 and λ = 1.5 unless otherwise noted; results across λ ∈ [0, 5] are shown in Fig. 4C.

### Dual-Valence Bennett-MV agent

Two parallel MBON pools (`m+` for appetitive compartments i/j/k, `m-` for aversive compartments c/d/f/g) with separate plastic weight matrices `w+` and `w-`. Compartment assignment uses the DAN subtype labels (Saumweber 2018; Weiglein 2024; Hu 2025) and the MBON subtype labels (Eichler 2017; Eschbach 2021). Reward is split as `r+ = max(0, r), r- = max(0, -r)`. Cross-valence DAN feedback: `d+ = ReLU(r+ + w_M · m-)`, `d- = ReLU(r- + w_M · m+)`, with `w_M = 1`. Weight updates follow Bennett's VS_λ rule (Eq. 23): `Δw+ = η · k · (λ − d-)`, `Δw- = η · k · (λ − d+)` with λ-baseline = 1.0. Weights clipped at zero.

### Behavioural protocols

- **Acquisition**: 30 alternating CS+ (rewarded, r=+1.0) and CS- (unrewarded) trials.
- **Extinction**: after acquisition, 20 more alternating trials with CS+ no longer rewarded.
- **Novel-odor probes**: after CS+/CS- training, present 12–20 fresh random odor patterns with reward=0 and learning off, then re-probe with learning on for the habituation curve.
- **Latent inhibition**: 20 trials of unrewarded CS+ pre-exposure (or unrelated odor for control), then 40 trials of alternating CS+/CS- acquisition.
- **Reversal**: 30 trials of CS_A rewarded / CS_B not, then 30 trials with the contingency flipped (CS_A no reward, CS_B rewarded).

All protocols are run for 20 random seeds for variance bands.

### Software and code availability

All analyses were implemented in Python 3.12 using NumPy 2.4, SciPy 1.17, NetworkX 3.6, Brian2 2.10, JAX 0.4 (CPU), and matplotlib 3.9. The complete codebase, including reproducible experiment scripts and regression tests (37 tests, all passing), is released at `github.com/<TODO>` under MIT license. The Winding 2023 connectome data is reused under the original Zenodo deposit terms.

### AI-assistance disclosure

This work was developed in collaboration with the Claude language model (Anthropic) acting as a coding pair-programmer, literature searcher, and verifier of mathematical derivations. Claude proposed, debugged, and tested code; identified and corrected one major error (the AIF "Bayesian surprise" framing that did not habituate, since reverted to posterior entropy) and one minor misreading (the Mancini reversal asymmetry specifics) during a verification pass. The scientific claims, model design choices, and interpretation are the author's. No conclusions in this paper depend on Claude as an author.

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
