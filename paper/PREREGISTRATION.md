# OSF Preregistration — DAN-c1 novelty habituation

**Draft URL:** https://osf.io/registries/drafts/6a0f719651897f1ece9da412/metadata
**Registry:** OSF Registries → OSF Preregistration template

Companion to the bioRxiv preprint *mb-dual-dan: A dual-pathway dopaminergic code for reward prediction and epistemic surprise in the larval Drosophila mushroom body* (Lyons, 2026; Zenodo concept DOI [10.5281/zenodo.20326853](https://doi.org/10.5281/zenodo.20326853); v0.1.1 DOI [10.5281/zenodo.20326854](https://doi.org/10.5281/zenodo.20326854)).

Paste these values into the corresponding OSF fields.

---

## Step 1 — Metadata

### Title

```
Pre-registered prediction: DAN-c1 in larval Drosophila habituates to novel odors prior to reinforcement
```

### Description

```
This preregistration time-stamps a falsifiable prediction made in the bioRxiv preprint "mb-dual-dan: A dual-pathway dopaminergic code for reward prediction and epistemic surprise in the larval Drosophila mushroom body" (Lyons, 2026; Zenodo DOI 10.5281/zenodo.20326853; GitHub https://github.com/Lyons800/mb-dual-dan).

The dual-pathway model proposes that the larval mushroom body uses two anatomically segregated dopaminergic systems: (i) reinforcement DANs (DAN-d1/f1/g1 for aversive valence, DAN-i1/j1/k1 for appetitive valence) that fire to predict-error signals during conditioning, and (ii) an epistemic DAN (DAN-c1, the anatomical analogue of adult PPL1-alpha'3) that fires to novelty independent of reward. A core prediction is that DAN-c1 should exhibit calcium-response habituation across repeated novel-odor presentations BEFORE any reinforcement is paired.

Predicted result (registered before experimental data are collected): in calcium-imaging recordings of DAN-c1 during 10 sequential presentations of a novel odor (no reward, no shock), DAN-c1 will show a monotonic decrease in ΔF/F of at least 30% from presentation 1 to presentation 10, whereas reinforcement-paired DANs (any of d1/f1/g1/i1/j1/k1) will show flat or non-decreasing responses across the same 10 presentations.

Falsification criteria: if DAN-c1 shows a flat or increasing response across the 10 novel-odor presentations (slope of ΔF/F vs. trial number not significantly negative at p<0.05), the dual-pathway model's epistemic-DAN channel is refuted as currently formulated.
```

### Contributors

Just Oisin Lyons (Administrator).

---

## Step 2 — Overview

### Hypothesis

```
H1 (primary): DAN-c1 calcium responses (ΔF/F) decrease monotonically across 10 sequential novel-odor presentations in the absence of any reinforcement, with mean reduction ≥30% from trial 1 to trial 10.

H2 (control): During the same 10 unreinforced trials, the six reinforcement-paired DANs (DAN-d1, DAN-f1, DAN-g1, DAN-i1, DAN-j1, DAN-k1) show no significant negative trend in ΔF/F across trials.

Together, H1 and H2 establish that novelty-habituation is selective to DAN-c1 and not a general property of mushroom-body DANs.
```

### Theoretical motivation

```
The dual-pathway model treats DAN-c1 as the larval analogue of adult PPL1-alpha'3, which Hattori et al. 2017 showed encodes a novelty signal in the adult fly. Bennett-style RPE models account for reinforcement-driven plasticity in MBONs but cannot explain latent inhibition (suppression of learning to pre-exposed odors; Jacob et al. 2021) without an additional novelty-coding channel. Our model derives latent inhibition computationally by gating the learning rate with a habituating posterior-entropy signal carried by DAN-c1. The empirical test below is the most direct possible falsification of the channel's existence.
```

---

## Step 3 — Research Design

### Study type

Confirmatory.

### Design

```
Open-loop in-vivo calcium imaging in head-fixed second-instar larvae expressing GCaMP7s in DAN-c1 (driver: SS00864-GAL4 or equivalent split-GAL4 line specific to DAN-c1; reporter: 20XUAS-IVS-GCaMP7s).

Each animal receives 10 sequential 5-second pulses of a novel odor (3-octanol, 10^-3 dilution in mineral oil) at 60-second inter-trial intervals, with no reinforcement (no fictive shock, no fictive sugar).

In a separate cohort, the same protocol is run with the GAL4 driver targeting one of DAN-d1/f1/g1/i1/j1/k1 (control DANs). Order of cohorts counterbalanced across imaging days.
```

### No-confounders statement

```
The 60s ITI is chosen to be long enough to wash out odor adaptation at the periphery (per Larsch et al. 2013 timescale ~10s) but short enough that within-session adaptation of the imaging preparation is minimal.
```

---

## Step 4 — Sampling

### Sample size

```
N = 12 animals per DAN target (DAN-c1, DAN-d1, DAN-f1, DAN-g1, DAN-i1, DAN-j1, DAN-k1) = 84 animals total. Sample size chosen by power analysis assuming a 30% mean ΔF/F reduction with within-animal SD ≈ 15% (typical for larval GCaMP7s in MB DANs per Vogt et al. 2021), giving power > 0.95 to detect H1 at α=0.05 (one-sided, paired across trials within animal).
```

### Stopping rule

```
Imaging session terminates after the 10th odor pulse for each animal, regardless of response. Animals are not excluded post-hoc except for the pre-registered exclusion criteria below.
```

### Exclusion criteria

```
1. Movement artifact exceeding 5 µm lateral shift uncorrectable by post-hoc registration.
2. Baseline ΔF/F drift exceeding ±20% across the 10 trials (preparation health indicator).
3. Failed expression check (GCaMP signal in target DAN below 2x autofluorescence floor in trial-1 baseline window).
```

---

## Step 5 — Variables

### Primary dependent variable

```
ΔF/F at the DAN-c1 (or control-DAN) soma, measured as peak fluorescence in the 0.5-3.0s window after odor onset, normalised to the 5-second pre-odor baseline. Computed per trial per animal.
```

### Primary independent variable

```
Trial number (1-10), treated as a within-animal continuous predictor.
```

### Secondary variables

```
- Time-to-peak ΔF/F (latency)
- Half-decay time post-peak
- Inter-trial baseline ΔF/F (excluded from primary analysis but reported)
```

---

## Step 6 — Analysis Plan

### Primary analysis

```
Linear mixed-effects model: ΔF/F ~ trial + (trial | animal_id), fit per DAN-target group using lme4 in R or statsmodels in Python.

H1 test: in the DAN-c1 group, is the fixed-effect slope of trial significantly negative at p<0.05 (one-sided)? Magnitude check: does the model-estimated mean ΔF/F at trial 10 fall ≥30% below the mean at trial 1?

H2 test: in each of the 6 control-DAN groups, is the fixed-effect slope of trial NOT significantly negative at p<0.05? Bonferroni-corrected across the 6 control groups.
```

### Decision rules

```
- H1 met AND H2 met (selective habituation in DAN-c1): the dual-pathway model's epistemic-DAN channel is supported.
- H1 NOT met (DAN-c1 does not habituate): the channel is refuted as formulated; the model must be revised.
- H1 met AND H2 violated (multiple DANs habituate): novelty habituation is not selective to DAN-c1; the channel exists but the assignment to DAN-c1 specifically is wrong.
```

### Robustness analyses (planned, not deciding)

```
1. Drop trial 1 (often anomalously large response from imaging-onset artifact) and refit.
2. Non-parametric Spearman ρ between trial number and ΔF/F per animal, aggregated by Stouffer's method.
3. Bayes-factor analogue: BF_10 for slope ≠ 0 using brms with default priors.
```

---

## Step 7 — Other

### Links

```
Preprint:    https://github.com/Lyons800/mb-dual-dan (pending bioRxiv DOI on submission)
Code:        https://github.com/Lyons800/mb-dual-dan
Zenodo:      https://doi.org/10.5281/zenodo.20326853 (concept DOI, latest version)
v0.1.1 DOI:  https://doi.org/10.5281/zenodo.20326854
```

### Conflicts of interest

```
None.
```

### Funding

```
None.
```

### Notes on scope

```
The model and falsifiable prediction were derived computationally (in silico) on the Winding et al. 2023 larval connectome before this preregistration. This document timestamps the prediction prior to the wet-lab calcium-imaging experiment that would test it. The author has no wet-lab capability at present and invites collaborators with larval imaging setups (Schleyer/Gerber, Cohn, Aso labs) to test the prediction.
```

---

## Step 8 — Review

Once steps 1-7 are filled, click **Review**, verify, then **Register**.

The registration is then time-stamped and immutable. The OSF DOI will be of the form `10.17605/OSF.IO/XXXXX` — add it to the manuscript's References before bioRxiv submission.
