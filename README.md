# brain

**Mushroom body as active inference vs reinforcement-prediction-error: a connectome-grounded comparison on the larval *Drosophila* mushroom body.**

## The thesis

The mushroom body has been modeled for thirty years as a reinforcement-prediction-error (RPE) machine — Bennett et al. 2021 *Nat Comms* is the canonical formulation. We test whether an active-inference (AIF) formulation, where dopamine neurons encode *expected free energy* (epistemic + pragmatic value) rather than reward prediction error, gives a better account of mushroom-body computation when the generative model's structural prior comes from the real connectome.

**Falsifiable claim:** during novel-odor probing, RPE predicts ~zero DAN response; AIF predicts a positive DAN transient proportional to posterior entropy H[s|o]. Anchored against Hattori 2017 α'3-compartment novelty data.

## Substrate

- **Connectome:** Winding et al. 2023 *Science* — full larval *Drosophila* (3,016 neurons, ~548k synapses), MB labeled (~240 KCs, 24 MBONs, 7–8 DANs).
- **Body:** larvaworld 2.2+ (purpose-built 2D Box2D larval body, olfactory + touch sensors).
- **Biophysics:** Brian2.
- **Inference:** pymdp 1.0 with factorized `A_dependencies` (no CPT explosion).
- **Baseline:** fork of `BrainsOnBoard/paper_RPEs_in_drosophila_mb` (Bennett 2021 MV model).

## Status

Phase 0: plumbing — in progress.

## Setup

```sh
# requires: macOS, Homebrew, swig (brew install swig)
uv sync
source .venv/bin/activate
python experiments/00_load_connectome.py
```

## Layout

```
src/brain/connectome/   loading + MB extraction + NT signs
src/brain/models/       rpe_baseline (Bennett), aif_agent (pymdp), shared
src/brain/tasks/        conditioning, extinction, novel_odor, mixtures
src/brain/embodiment/   larvaworld integration
experiments/            runnable scripts
data/winding_2023/      connectome data (downloaded, gitignored)
```
