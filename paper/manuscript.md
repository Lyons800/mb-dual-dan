# A dual-pathway dopaminergic code for reward prediction and epistemic surprise in the larval *Drosophila* mushroom body

**Author:** Oisin Lyons. **AI-assistance disclosure:** see Methods.

---

## Abstract

The *Drosophila* mushroom body (MB) has been modeled for three decades as a single-channel reward-prediction-error (RPE) machine. Yet dopamine neurons (DANs) fire to novel odors before any reward (Hattori 2017), prior exposure produces latent inhibition (Jacob et al. 2021), and contingency reversal completes within one cycle (Mancini 2019) — phenomena single-channel RPE structurally cannot explain. The newly released larval connectome (Winding 2023; 3,016 neurons, every synapse mapped) lets us isolate which architectural ingredients each phenomenon needs by holding wiring fixed and varying only the learning rule.

We compare four agents on the same connectome substrate (588-neuron MB subgraph; 144 KCs, 48 MBONs in 24 bilateral types, 14 DAN cells in 7 types). Each ingredient is **individually necessary**: signed-weight RPE for extinction; a Bayesian surprisal channel (DAN ∝ posterior entropy / −log p(KC)) for novelty signaling and habituation; precision-weighted learning rate (η_eff = η_0(1+λ·surprisal)) for latent inhibition; parallel appetitive/aversive traces with cross-valence DAN feedback (Bennett-MV / Felsenberg-POM) for reversal at acquisition speed (matching the qualitative parallel-opposing-memories signature; the specific 3:1 ratio of Mancini 2019 requires asymmetric appetitive-vs-aversive plasticity, future work). Removing any one breaks at least one signature. Unsupervised clustering of MBINs by shared KC pools recovers Eichler 2017's compartmental organisation from wiring alone (canonical DAN-i1 ↔ MBON-i1 pair, J=0.82). We predict: the larval DAN encoding perceptual surprise should habituate to novel odors before reinforcement — testable by calcium imaging in DAN-c1.

---

## Introduction

The *Drosophila* mushroom body is the best-characterised associative-learning circuit in any animal. Three decades of experiment converge on a stereotyped picture: olfactory projection neurons (PNs) feed a sparse code in ~2,000 Kenyon cells (KCs), whose synapses onto a few dozen mushroom body output neurons (MBONs) are bidirectionally modulated by dopaminergic teaching signals from ~20 DAN types, each occupying its own compartment (Tully & Quinn 1985; Heisenberg 2003; Aso et al. 2014; Modi, Shuai & Turner 2020). The leading computational account is Bennett et al. 2021's reinforcement-prediction-error (RPE) model: a DAN encodes `r − m̂` at each compartment, KC→MBON synapses adapt by a three-factor rule, and extinction emerges naturally from MBON-driven prediction error feedback (Felsenberg et al. 2018). This single-channel picture has dominated the field for a decade.

Yet several robust observations remain unaccounted for. Hattori et al. 2017 reported that the adult PPL1-α'3 DAN fires to novel odors *before any reward pairing* and habituates with re-exposure — a signal pure RPE cannot generate, because there is no reward to be wrong about. Jacob et al. 2021 documented latent inhibition: pre-exposure to an odor without reward measurably slows subsequent CS+US learning, requiring some attention-like gating that flat RPE lacks. Mancini et al. 2019 showed larval reversal completes in roughly one cycle compared to three for the original acquisition — an asymmetry incompatible with single-trace gradient learning, which is symmetric in η. Each observation has been treated as a curious exception. We argue they are diagnostic.

The recent publication of the complete larval *Drosophila* synaptic-resolution connectome (Winding et al. 2023) makes it possible to test which architectural ingredients each of these signatures requires, by holding the substrate fixed and varying only the learning rule. Using the 588-neuron MB subgraph as a common substrate, we compare four agents — classical RPE, a Bayesian-observer "active inference" channel, a precision-weighted hybrid, and a full dual-valence model — across four behavioural protocols. Each architectural ingredient turns out to be individually necessary for a specific phenomenon. The framework also yields a concrete falsifiable prediction at the cellular level: DAN-c1 should carry a perceptual-surprise component analogous to adult PPL1-α'3.

---

## Results

### A connectome substrate that reveals compartment structure (Fig. 1)

We extracted the larval MB subgraph from Winding 2023 by filtering on the cell-type annotations: 144 Kenyon cells (KCs), 48 MBONs (24 cell types in left/right pairs), 14 DAN cells (7 types — DAN-c1/d1/f1/g1 aversive and DAN-i1/j1/k1 appetitive per Saumweber 2018, Weiglein 2024 and Hu 2025), 4 OAN cells, and 206 projection neurons — 588 neurons connected by 22,331 synaptic edges (Fig. 1A). DAN-h1, present in older anatomical descriptions, is absent from the L1 EM volume (Eichler 2017 confirms this developmental absence).

To test whether the wiring alone encodes the canonical compartmental organisation of Eichler 2017, we computed the bipartite Jaccard similarity between each DAN and each MBON on their shared KC pools (Fig. 1B), then thresholded and hierarchically reordered. The block-diagonal structure is immediately visible. The single highest-overlap pair is DAN-i1 ↔ MBON-i1 with J = 0.82 (Fig. 1C); the same pattern repeats across hemispheres. **Anatomical compartment identity is recoverable from synaptic wiring alone.** This validates the substrate for the modelling that follows.

Onto this substrate we built four agents (Fig. 1D), each sharing the same PN→KC sparse coder (top-k=7 of 144 active per odor, consistent with the optimal-sparsity target derived by Litwin-Kumar et al. 2017), and each differing only in the KC→MBON learning rule and the DAN signal. The four are: (i) classical RPE (Bennett 2021 single-valence with signed weights), (ii) a Bayesian-observer "AIF" agent tracking posterior entropy and surprisal, (iii) a precision-weighted hybrid `η_eff = η_0 (1 + λ·surprisal)`, and (iv) a full Bennett-MV dual-valence model with parallel appetitive and aversive traces. All four are tested on identical behavioural protocols.

![**Figure 1.** Substrate and architecture. (A) MB subgraph composition (Winding 2023). (B) DAN × MBON bipartite Jaccard heatmap on shared KC pools, hierarchically reordered — block structure reveals compartments without anatomical labels. (C) Highest-overlap pairs; DAN-i1 ↔ MBON-i1 (J=0.82) and other within-compartment matches marked with asterisks. (D) Four agents share the connectome substrate and PN→KC sparse coder; only the KC→MBON learning rule and DAN signal differ.](figures/fig1_substrate.png)

### Signed-weight RPE supports acquisition and extinction (Fig. 2)

On a standard alternating CS+/CS- conditioning task with reward paired to CS+, the RPE agent acquires the appetitive response cleanly: across 20 random seeds, m_hat for CS+ rises from 0.20 ± 0.02 to 0.94 ± 0.01 over 30 CS+ presentations (60 total alternating trials), with CS- staying flat near zero (Fig. 2A). The DAN error (r − m_hat) decays from 0.82 to 0.06 (Fig. 2B), confirming the standard Rescorla-Wagner trajectory. When the contingency is reversed (CS+ → no reward), the model extinguishes 0.94 → 0.15 over 20 trials (Fig. 2C) — but only because signed weights are permitted. Clipping weights at zero (a common implementation choice we removed in our rigor pass) makes extinction structurally impossible: w+ can only grow, never shrink. **Signed-weight RPE is necessary for acquisition and extinction.**

![**Figure 2.** RPE on the connectome substrate. (A) Acquisition curve with N=20 seed variance bands; CS+ m_hat rises while CS- remains flat. (B) DAN error r−m_hat decays to ~0.06 by trial 30. (C) Acquisition → extinction; signed weights enable the m_hat 0.94 → 0.15 collapse during extinction. Without signed weights, extinction is structurally impossible.](figures/fig2_rpe.png)

### A Bayesian surprisal channel is necessary for novelty signalling (Fig. 3)

After training, we presented familiar and novel-odor probes without reward (no learning during probe phase). The RPE agent's |DAN| on familiar CS+ is large (∼0.5 on average due to extinction-style PE when reward is omitted) but contains no novelty signal — its DAN magnitude is bounded by learned m_hat alone (Fig. 3A). The AIF agent shows the opposite profile: confident posterior (entropy ≈ 0) on familiar odors, but **bimodal** novelty response on 200 unique probes (Fig. 3B) — approximately half cluster at zero (novel KC patterns that happen to align with one trained class) and approximately half cluster at log(2) ≈ 0.69 (the theoretical maximum entropy for two classes, achieved when the KC pattern strongly mixes features of both). Repeated exposure to a single novel odor with learning enabled produces immediate habituation: AIF DAN drops from log(2) on the first trial to 0 on the second (Fig. 3C), reproducing the canonical Hattori 2017 α'3 novelty-habituation profile. **A Bayesian surprisal channel — explicitly tracking posterior entropy or model-evidence surprisal − log p(KC) − is necessary for these signatures.**

![**Figure 3.** A Bayesian surprisal channel for novelty. (A) Mean |DAN| on familiar vs novel probes for the two single-channel models. RPE's |DAN| on familiar is dominated by extinction-style PE; AIF is confident (entropy ≈ 0). (B) Distribution of AIF DAN across 200 novel-odor probes: bimodal at 0 and log(2), reflecting class-match and truly-ambiguous novel patterns. (C) Habituation curve on repeated exposure to a single novel odor — drops from log(2) to ~0 in one trial.](figures/fig3_novel_odor.png)

### Precision-weighted learning yields latent inhibition (Fig. 4)

Pre-exposure to an odor without reward is known to slow subsequent CS+US learning (latent inhibition, Jacob et al. 2021). The mechanistic prediction of any precision-weighted theory is direct: pre-exposure habituates surprisal, lowering η_eff at the moment when reward is later introduced. We tested this with a two-group design: pre-exposed (20 unrewarded CS+ presentations) vs. control (20 unrelated-odor presentations), each followed by 40-trial CS+/CS- acquisition (Fig. 4A,B). Across 20 seeds, RPE shows essentially overlapping acquisition curves between groups — the structural prediction for a model with no precision-weighting mechanism. The Dual-DAN model shows a dramatic separation: the control group reaches m_hat = 1.0 by trial 1, while the pre-exposed group ramps slowly from 0 to ∼0.8 over 40 trials. A λ-sweep (Fig. 4C) confirms the causal mechanism: the LI effect rises monotonically from 0.06 at λ = 0 (the pure-RPE limit) to 0.89 at λ = 5. **Precision-weighted learning rate is necessary for latent inhibition.**

![**Figure 4.** Latent inhibition from precision-weighted learning. (A) RPE shows essentially overlapping acquisition curves between control and pre-exposed groups (N=20 seeds) — structural no-LI. (B) Dual-DAN shows dramatic separation: control acquires CS+ in 1 trial; pre-exposed ramps slowly over 40 trials. (C) λ dose-response: LI effect (AUC_ctrl − AUC_pre) rises monotonically from 0.06 at λ=0 (RPE limit) to 0.89 at λ=5.](figures/fig4_latent_inhibition.png)

### Parallel valence traces enable fast reversal (Fig. 5)

The Mancini 2019 reversal protocol exposes a fourth phenomenon that single-valence models cannot capture. After 30 CS_A → reward and 30 CS_B → nothing acquisition trials, the contingency reverses (CS_A → nothing, CS_B → reward) for 30 more trials. To compare all four agents in the same dynamical regime, we set η = 0.005 for every model so that acquisition takes a non-trivial number of trials and reversal kinetics are visible (the qualitative result is robust across η ∈ [0.001, 0.025]; see robustness sweep in Methods). The four models diverge sharply (Fig. 5A–D): pure RPE reverses symmetrically (the single trace and identical η govern both directions equally); AIF fails entirely (accumulated Beta-Bernoulli evidence locks the class assignment for 50+ reversal trials); the Dual-DAN hybrid reverses but slowly (the surprisal boost is gone during reversal because both patterns are now familiar); only the **Dual-Valence** Bennett-MV model reverses cleanly via Felsenberg 2018's parallel-opposing-memories mechanism, visible in the trace decomposition inset (Fig. 5D). The trace decomposition makes the mechanism explicit: when CS_A flips from rewarded to unrewarded, m+(CS_A) persists at the acquired level — the original appetitive trace is **not erased** — but m-(CS_A) grows in parallel, driven by the cross-valence DAN feedback d- = ReLU(w_M · m+) firing on the now-unrewarded probe. Net m_hat = m+ − m- collapses fast because two things are moving simultaneously instead of one. **Cross-valence DAN feedback with parallel appetitive/aversive traces is necessary for reversal at acquisition speed.** Quantitatively matching Mancini's specific 3:1 acquisition-vs-reversal ratio requires asymmetric appetitive-vs-aversive plasticity parameters (consistent with Davis 2023's observation that aversive memories are more stable), which we leave for future work.

![**Figure 5.** Reversal learning across four models. (A) RPE: slow symmetric reversal. (B) AIF: cannot reverse — m_hat(CS_A) stays at 1.0 for 50+ reversal trials. (C) Dual-Valence Bennett-MV: fast reversal via Felsenberg POM. (D) Mechanism: m+ (appetitive trace, teal) persists during reversal while m- (aversive trace, red) grows in parallel; m_hat = m+ − m- (dashed) collapses fast because two things move simultaneously.](figures/fig5_reversal.png)

### Necessity is jointly satisfied, not pairwise

Across the four protocols, no subset of three architectural ingredients suffices: signed-weight RPE fails on novelty and latent inhibition; surprisal-channel-only fails on extinction and reversal; precision-weighting alone fails on reversal; parallel valence traces alone fail on latent inhibition. **All four ingredients are individually necessary, and the four together are sufficient for the four signatures tested.** The connectome substrate does not pick the learning rule — behaviour does.

---

## Discussion

We have shown that four architectural ingredients are jointly necessary to reproduce four well-established behavioural phenomena in the larval *Drosophila* mushroom body, holding the synaptic substrate constant at the Winding 2023 connectome. No subset of three ingredients suffices. The result resolves a tension that has accumulated in the field: the standard single-channel RPE account (Bennett 2021 and predecessors) is correct for acquisition and extinction, but a separate channel is needed for novelty-related phenomena (Hattori 2017), precision-weighting is needed for latent inhibition (Jacob et al. 2021), and parallel valence traces are needed for fast reversal (Mancini 2019; Felsenberg 2018). Each phenomenon has been treated as a curious exception. We argue they share a common diagnosis: cortex-like learning rules layered on the same connectome substrate.

The DAN-c1 prediction is the most directly falsifiable contribution. Saumweber 2018 showed that direct optogenetic activation of DAN-c1 alone produces no associative memory — anomalous for a DAN under standard RPE accounts. Hu 2025 has since documented an olfactory aversive learning role for DAN-c1 via D2R autoregulation. These are not mutually exclusive in a multi-signal DAN framework (Engelhard 2019; Dabney 2020), and we hypothesise that DAN-c1 also carries a perceptual-surprise component analogous to adult PPL1-α'3 (Hattori 2017). The empirical test is direct: calcium imaging of DAN-c1 during the first exposure to a novel odor, before any reward pairing, should show a transient activation that habituates with repeated exposure. A negative result would falsify the specific cellular hypothesis without affecting the architectural claim — any other DAN, or any FBN, could in principle carry the surprisal channel. The prediction is therefore both specific and recoverable.

Our model identifies one quantitative gap: the symmetric Bennett-MV formulation reproduces the qualitative POM signature (reversal as fast as acquisition) but not Mancini's specific 3:1 acquisition-vs-reversal ratio. We achieve a ratio near 1 across the tested η range; the data require closer to 3. This is informative. The Davis lab has documented asymmetric stability between appetitive and aversive memories at the cellular level (different DAN receptors, distinct decay constants, ARM/ASM separation). Implementing this asymmetry — η+ ≠ η-, λ+ ≠ λ-, or distinct decay rates — should close the quantitative gap, but is a parameter and not an architectural change. The prediction is testable: aversive memory traces should retain through longer intervals than appetitive ones in computational terms exactly as Davis 2023 reports them biologically.

Our approach contrasts with three adjacent lines of work. The Nawrot lab (Jurgensen 2024, Rachad 2025) has the closest published substrate — a connectome-informed spiking larval MB with RPE-only DAN signalling. They reproduce delay conditioning and second-order conditioning but do not include a perceptual-surprise channel and have not been shown to reproduce latent inhibition. Jiang & Litwin-Kumar 2021 argued for heterogeneous DAN coding without proposing an explicit AIF/surprisal channel. Active-inference theorists (Friston, Pezzulo, Lanillos, Schwartenbeck) have argued for precision-weighting and EFE-based policy selection but have not applied these to MB substrates. The contribution here is the synthesis on a real connectome and the demonstration that all three signals are required.

Several limitations should be acknowledged. The "AIF" agent in this paper is a perception-only Bayesian observer rather than a full active-inference agent — there is no action selection by EFE minimisation, no policy precision γ, and the latent state is supervised by reward outcome rather than truly latent. A complete AIF treatment using pymdp 1.0 or RxInfer.jl is a natural next step. The MBON pooling collapses 48 cells into two valence populations (per Eichler 2017 / Eschbach 2021 single-MBON behavioural labels); per-compartment plasticity with the full eligibility-trace 3-factor rule (Aso 2014/2016, Cohn 2015) would be a faithful refinement. The substrate is L1; scaling to adult FlyWire (2,580 KCs, 19 MBON types) is the obvious extension and is now feasible. Finally, the agent is disembodied — closing the sensorimotor loop in NeuroMechFly v2 / FlyGym v2 / larvaworld would let us test behavioural outputs against published navigation data.

We close with the observation that the larval *Drosophila* MB is now the smallest brain region in any animal where every synapse is mapped, multiple behavioural protocols have been quantitatively characterised, and individual DANs and MBONs can be perturbed optogenetically. This makes it an ideal proving ground for *which* learning rules biology has converged on. Our claim is not that the brain runs precisely the equations we describe; it is that any complete model of MB function must accommodate the four ingredients we have shown to be individually necessary. We expect this constraint to sharpen as adult connectomes mature and as more behavioural protocols are quantitatively characterised on the same substrate.

---

## Methods

### Connectome substrate

We used the larval *Drosophila* synaptic-resolution connectome from Winding et al. 2023 (*Science*), Supplementary Data S1 (publicly released, no restrictions). The all-to-all connectivity matrix (2,952 × 2,952 neurons; 110,677 nonzero edges; 352,611 synaptic contacts) was transposed at load time to the standard `W[post, pre]` convention so that `y = W · x` gives postsynaptic activity. The accompanying `annotations.csv` provides cell-type labels and subtype annotations (e.g. DAN-c1, MBON-i1) that we use throughout.

The MB subgraph is extracted by filtering on the cell-type column to retain KC, MBON, MBIN (the supercategory containing DANs, OANs, and other modulatory cells), MB-FBN (feedback neurons), MB-FFN (feedforward neurons), and PN (projection neurons). DAN identity is recovered by parsing the `additional_annotations` subtype labels matching the pattern `DAN-*`. The resulting subgraph contains 588 neurons (144 KCs, 48 MBONs, 28 MBINs split into 14 DANs / 4 OANs / 10 other-modulatory cells, 108 MB-FBNs, 54 MB-FFNs, 206 PNs) connected by 22,331 synaptic edges (~157k total synaptic contacts).

Neurotransmitter signs are not in the Winding S1 dataset and the adult-FAFB neurotransmitter classifier of Eckstein et al. 2024 does not apply to L1 larval EM. We use cell-class conventions: KC, MBON, PN, and sensory neurons are treated as cholinergic (excitatory); LN and APL as GABAergic (inhibitory); MBINs are treated as modulatory with sign +1 in the linear readout (their teaching action is implemented separately in the plasticity rule, not as fast postsynaptic drive).

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

### Data and code availability

All analyses were implemented in Python 3.12 using NumPy 2.4, SciPy 1.17, NetworkX 3.6, Brian2 2.10, JAX 0.4 (CPU), and matplotlib 3.9. The complete codebase, including reproducible experiment scripts and regression tests (37 tests, all passing), is released at <https://github.com/Lyons800/mb-dual-dan> under MIT license and archived on Zenodo (concept DOI [10.5281/zenodo.20326853](https://doi.org/10.5281/zenodo.20326853); this paper analyses v0.1.1, DOI [10.5281/zenodo.20326854](https://doi.org/10.5281/zenodo.20326854)). The Winding et al. 2023 larval connectome (Supplementary Data S1) is reused without modification under the original Zenodo deposit terms (zenodo.org/records/10676866). No new data are generated by this work; the manuscript is entirely *in silico* analysis of public data. Figures are deterministic at the random seeds specified in each script.

### Author contributions

OL conceived the project, wrote all code, performed all analyses, and drafted the manuscript. AI-assistance disclosure below.

### Funding

This work received no external funding.

### Competing interests

The author declares no competing interests.

### AI-assistance disclosure

This work was developed in collaboration with the Claude language model (Anthropic) acting as a coding pair-programmer, literature searcher, and verifier of mathematical derivations. Claude proposed, debugged, and tested code; identified and corrected one major error (the AIF "Bayesian surprise" framing that did not habituate, since reverted to posterior entropy) and one minor misreading (the Mancini reversal asymmetry specifics) during a verification pass. The scientific claims, model design choices, and interpretation are the author's. No conclusions in this paper depend on Claude as an author.

---

## References

Aso Y, Hattori D, Yu Y, Johnston RM, Iyer NA, Ngo TB, Dionne H, Abbott LF, Axel R, Tanimoto H, Rubin GM. 2014. The neuronal architecture of the mushroom body provides a logic for associative learning. *eLife* 3:e04577. doi:10.7554/eLife.04577

Aso Y, Sitaraman D, Ichinose T, Kaun KR, Vogt K, Belliart-Guerin G, Placais P-Y, Robie AA, Yamagata N, Schnaitmann C, Rowell WJ, Johnston RM, Ngo TB, Chen N, Korff W, Nitabach MN, Heberlein U, Preat T, Branson KM, Tanimoto H, Rubin GM. 2014. Mushroom body output neurons encode valence and guide memory-based action selection in *Drosophila*. *eLife* 3:e04580. doi:10.7554/eLife.04580

Aso Y, Rubin GM. 2016. Dopaminergic neurons write and update memories with cell-type-specific rules. *eLife* 5:e16135. doi:10.7554/eLife.16135

Bennett JEM, Philippides A, Nowotny T. 2021. Learning with reinforcement prediction errors in a model of the *Drosophila* mushroom body. *Nature Communications* 12:2569. doi:10.1038/s41467-021-22592-4

Cohn R, Morantte I, Ruta V. 2015. Coordinated and compartmentalized neuromodulation shapes sensory processing in *Drosophila*. *Cell* 163:1742–1755. doi:10.1016/j.cell.2015.11.019

Dabney W, Kurth-Nelson Z, Uchida N, Starkweather CK, Hassabis D, Munos R, Botvinick M. 2020. A distributional code for value in dopamine-based reinforcement learning. *Nature* 577:671–675. doi:10.1038/s41586-019-1924-6

Davis RL. 2023. Learning and memory using *Drosophila melanogaster*: a focus on advances made in the fifth decade of research. *Genetics* 224:iyad085. doi:10.1093/genetics/iyad085

Eckstein N, Bates AS, Champion A, Du M, Yin Y, Schlegel P, Lu A K-Y, Rymer T, Finley-May S, Paterson T, Parekh R, Dorkenwald S, Matsliah A, Yu S-c, McKellar C, Sterling A, Eichler K, Costa M, Seung S, Murthy M, Hartenstein V, Jefferis GSXE, Funke J. 2024. Neurotransmitter classification from electron microscopy images at synaptic sites in *Drosophila melanogaster*. *Cell* 187:2574–2594. doi:10.1016/j.cell.2024.03.016

Eichler K, Li F, Litwin-Kumar A, Park Y, Andrade I, Schneider-Mizell CM, Saumweber T, Huser A, Eschbach C, Gerber B, Fetter RD, Truman JW, Priebe CE, Abbott LF, Thum AS, Zlatic M, Cardona A. 2017. The complete connectome of a learning and memory centre in an insect brain. *Nature* 548:175–182. doi:10.1038/nature23455

Engelhard B, Finkelstein J, Cox J, Fleming W, Jang HJ, Ornelas S, Koay SA, Thiberge SY, Daw ND, Tank DW, Witten IB. 2019. Specialized coding of sensory, motor and cognitive variables in VTA dopamine neurons. *Nature* 570:509–513. doi:10.1038/s41586-019-1261-9

Eschbach C, Fushiki A, Winding M, Schneider-Mizell CM, Shao M, Arruda R, Eichler K, Valdes-Aleman J, Ohyama T, Thum AS, Gerber B, Fetter RD, Truman JW, Litwin-Kumar A, Cardona A, Zlatic M. 2020. Recurrent architecture for adaptive regulation of learning in the insect brain. *Nature Neuroscience* 23:544–555. doi:10.1038/s41593-020-0607-9

Eschbach C, Fushiki A, Winding M, Afonso B, Andrade IV, Cocanougher BT, Eichler K, Gepner R, Si G, Valdes-Aleman J, Fetter RD, Gershow M, Jefferis GSXE, Samuel ADT, Truman JW, Cardona A, Zlatic M. 2021. Circuits for integrating learned and innate valences in the insect brain. *eLife* 10:e62567. doi:10.7554/eLife.62567

Felsenberg J, Barnstedt O, Cognigni P, Lin S, Waddell S. 2017. Re-evaluation of learned information in *Drosophila*. *Nature* 544:240–244. doi:10.1038/nature21716

Felsenberg J, Jacob PF, Walker T, Barnstedt O, Edmondson-Stait AJ, Pleijzier MW, Otto N, Schlegel P, Sharifi N, Perisse E, Smith CS, Lauritzen JS, Costa M, Jefferis GSXE, Bock DD, Waddell S. 2018. Integration of parallel opposing memories underlies memory extinction. *Cell* 175:709–722.e15. doi:10.1016/j.cell.2018.08.021

Friston K. 2010. The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11:127–138. doi:10.1038/nrn2787

Friston K, FitzGerald T, Rigoli F, Schwartenbeck P, Pezzulo G. 2017. Active inference: a process theory. *Neural Computation* 29:1–49. doi:10.1162/NECO_a_00912

Hattori D, Aso Y, Swartz KJ, Rubin GM, Abbott LF, Axel R. 2017. Representations of novelty and familiarity in a mushroom body compartment. *Cell* 169:956–969.e17. doi:10.1016/j.cell.2017.04.028

Heisenberg M. 2003. Mushroom body memoir: from maps to models. *Nature Reviews Neuroscience* 4:266–275. doi:10.1038/nrn1074

Hige T, Aso Y, Modi MN, Rubin GM, Turner GC. 2015. Heterosynaptic plasticity underlies aversive olfactory learning in *Drosophila*. *Neuron* 88:985–998. doi:10.1016/j.neuron.2015.11.003

Hu Y, Wang C, Yang L, Pan G, Liu H, Yu G, Ye B. 2025. A pair of dopaminergic neurons DAN-c1 mediate *Drosophila* larval aversive olfactory learning through D2-like receptors. *eLife* 13:RP100890. doi:10.7554/eLife.100890

Itti L, Baldi P. 2009. Bayesian surprise attracts human attention. *Vision Research* 49:1295–1306. doi:10.1016/j.visres.2008.09.007

Jacob PF, Vargas-Gutierrez P, Okray Z, Vietti-Michelina S, Felsenberg J, Waddell S. 2021. Prior experience conditionally inhibits the expression of new learning in *Drosophila*. *Current Biology* 31:3490–3503.e3. doi:10.1016/j.cub.2021.05.056

Jiang L, Litwin-Kumar A. 2021. Models of heterogeneous dopamine signaling in an insect learning and memory center. *PLOS Computational Biology* 17:e1009205. doi:10.1371/journal.pcbi.1009205

Jurgensen A-M, Sakagiannis P, Schleyer M, Gerber B, Nawrot MP. 2024. Prediction error drives associative learning and conditioned behavior in a spiking model of *Drosophila* larva. *iScience* 27:108640. doi:10.1016/j.isci.2023.108640

Litwin-Kumar A, Harris KD, Axel R, Sompolinsky H, Abbott LF. 2017. Optimal degrees of synaptic connectivity. *Neuron* 93:1153–1164.e7. doi:10.1016/j.neuron.2017.01.030

Mancini N, Hranova S, Weber J, Weiglein A, Schleyer M, Weber D, Thum AS, Gerber B. 2019. Reversal learning in *Drosophila* larvae. *Learning & Memory* 26:424–435. doi:10.1101/lm.049510.119

Modi MN, Shuai Y, Turner GC. 2020. The *Drosophila* mushroom body: from architecture to algorithm in a learning circuit. *Annual Review of Neuroscience* 43:465–484. doi:10.1146/annurev-neuro-080317-062333

Parr T, Friston KJ. 2019. Generalised free energy and active inference. *Biological Cybernetics* 113:495–513. doi:10.1007/s00422-019-00805-w

Rachad EY, Deimel SH, Epple L, et al., Nawrot MP. 2025. Functional dissection of a neuronal brain circuit mediating higher-order associative learning. *Cell Reports* 44:115593. doi:10.1016/j.celrep.2025.115593

Saumweber T, Rohwedder A, Schleyer M, Eichler K, Chen Y-C, Aso Y, Cardona A, Eschbach C, Kobler O, Voigt A, Durairaja A, Mancini N, Zlatic M, Truman JW, Thum AS, Gerber B. 2018. Functional architecture of reward learning in mushroom body extrinsic neurons of larval *Drosophila*. *Nature Communications* 9:1104. doi:10.1038/s41467-018-03130-1

Schwartenbeck P, FitzGerald THB, Mathys C, Dolan R, Friston K. 2015. The dopaminergic midbrain encodes the expected certainty about desired outcomes. *Cerebral Cortex* 25:3434–3445. doi:10.1093/cercor/bhu159

Schwartenbeck P, Passecker J, Hauser TU, FitzGerald THB, Kronbichler M, Friston KJ. 2019. Computational mechanisms of curiosity and goal-directed exploration. *eLife* 8:e41703. doi:10.7554/eLife.41703

Sutton RS. 1988. Learning to predict by the methods of temporal differences. *Machine Learning* 3:9–44. doi:10.1007/BF00115009

Tully T, Quinn WG. 1985. Classical conditioning and retention in normal and mutant *Drosophila melanogaster*. *Journal of Comparative Physiology A* 157:263–277. doi:10.1007/BF01350033

Weiglein A, Thoener J, Feldbruegge I, Warzog L, Mancini N, Schleyer M, Gerber B. 2024. Four individually identified paired dopamine neurons signal taste punishment in larval *Drosophila*. *eLife* 12:RP91387. doi:10.7554/eLife.91387

Winding M, Pedigo BD, Barnes CL, Patsolic HG, Park Y, Kazimiers T, Fushiki A, Andrade IV, Khandelwal A, Valdes-Aleman J, Li F, Randel N, Barsotti E, Correia A, Fetter RD, Hartenstein V, Priebe CE, Vogelstein JT, Cardona A, Zlatic M. 2023. The connectome of an insect brain. *Science* 379:eadd9330. doi:10.1126/science.add9330

Yamada D, Bushey D, Li F, Hibbard KL, Sammons M, Funke J, Litwin-Kumar A, Hige T, Aso Y. 2023. Hierarchical architecture of dopaminergic circuits enables second-order conditioning in *Drosophila*. *eLife* 12:e79042. doi:10.7554/eLife.79042

Yu AJ, Dayan P. 2005. Uncertainty, neuromodulation, and attention. *Neuron* 46:681–692. doi:10.1016/j.neuron.2005.04.026
