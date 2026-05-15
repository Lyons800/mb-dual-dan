# mb-dual-dan

**A dual-pathway dopaminergic code for reward prediction and epistemic surprise in the larval *Drosophila* mushroom body.**

Companion code repository for the paper of the same name (manuscript in `paper/manuscript.md`).

## The claim

Four architectural ingredients — signed-weight RPE, a Bayesian surprisal channel, surprisal-modulated learning rate, and parallel valence traces — are *each individually necessary* for four classical mushroom-body phenomena (acquisition+extinction, novelty signaling, latent inhibition, fast reversal). All four run on the same Winding 2023 larval connectome substrate.

| Phenomenon | Required ingredient | Reference |
|---|---|---|
| Acquisition + extinction | Signed-weight RPE | Bennett 2021; Felsenberg 2018 |
| Novel-odor DAN firing + habituation | Bayesian surprisal channel | Hattori 2017 |
| Latent inhibition | Precision-weighted η_eff = η_0(1 + λ·σ) | Jacob & Waddell 2022 |
| Fast reversal | Parallel appetitive/aversive traces (Bennett-MV / Felsenberg-POM) | Mancini 2019 |

**Falsifiable prediction**: DAN-c1 (the larval analogue of adult PPL1-α'3) should habituate to novel odors before reinforcement, testable by calcium imaging.

## Setup

```sh
# Requires Python 3.12 and Homebrew (for swig, needed by box2d-py if you want larvaworld embodiment).
brew install swig
git clone https://github.com/Lyons800/mb-dual-dan.git
cd mb-dual-dan
uv sync
```

The Winding 2023 connectome is fetched automatically — see `data/winding_2023/README.md`.

## Reproducing the paper figures

```sh
.venv/bin/python paper/figures/fig1_substrate.py
.venv/bin/python paper/figures/fig2_rpe.py
.venv/bin/python paper/figures/fig3_novel_odor.py
.venv/bin/python paper/figures/fig4_latent_inhibition.py
.venv/bin/python paper/figures/fig5_reversal.py
```

All five figures regenerate from the connectome substrate; outputs are deterministic at the random seeds set in each script.

## Running the experiments end-to-end

```sh
.venv/bin/python -m pytest tests/                       # 37 regression tests
.venv/bin/python experiments/00_load_connectome.py      # substrate smoke test
.venv/bin/python experiments/01_rpe_baseline.py         # Bennett RPE acquisition
.venv/bin/python experiments/02_aif_agent.py            # head-to-head RPE vs AIF
.venv/bin/python experiments/03_novel_odor.py           # novel-odor probing
.venv/bin/python experiments/04_compartments.py         # bipartite Jaccard recovery
.venv/bin/python experiments/05_extinction.py           # extinction kinetics
.venv/bin/python experiments/06_latent_inhibition.py    # the LI signature
.venv/bin/python experiments/07_reversal.py             # 4-model reversal comparison
.venv/bin/python experiments/08_robustness.py           # multi-seed bands + sweeps
```

## Repository layout

```
brain/
├── paper/                              the manuscript + figures
│   ├── manuscript.md                   ~4,000 words, eLife-targeted
│   └── figures/                        Fig 1-5 + render scripts
├── src/brain/
│   ├── connectome/                     Winding 2023 loader, MB extractor,
│   │                                   compartment discovery, valence map
│   ├── models/                         RPE, AIF, Dual-DAN, Dual-Valence agents
│   ├── tasks/                          conditioning, extinction, novel-odor,
│   │                                   latent inhibition, reversal protocols
│   ├── embodiment/                     larvaworld integration (Phase 5, future)
│   └── robustness.py                   multi-seed runner + parameter sweeps
├── experiments/                        runnable end-to-end protocols
├── tests/                              37 regression tests
└── data/winding_2023/                  Winding 2023 Supplementary S1 (downloaded)
```

## Data and code provenance

- **Connectome**: Winding et al. 2023, *Science*. Supplementary Data S1 from <https://github.com/brain-networks/larval-drosophila-connectome>. Used under the original release terms.
- **DAN/MBON valence labels**: assembled from Saumweber 2018, Weiglein 2024, Hu 2025, Eschbach 2021. Compiled in `src/brain/connectome/valence.py`.
- **RPE baseline equations**: Bennett, Philippides & Nowotny 2021, *Nat Commun*. Single-valence reduction and full MV both implemented.
- **AIF / Bayesian observer**: standard Beta-Bernoulli naive Bayes with three readout signals (entropy, surprise, surprisal) — see `src/brain/models/aif_agent.py` for the explicit formulations and citations.

## AI-assistance disclosure

The codebase was developed in collaboration with the Claude language model (Anthropic) acting as a coding pair-programmer, literature searcher, and verifier of equations. Claude proposed, debugged, and tested code; identified and corrected one major formulation error (AIF "Bayesian surprise" reverted to posterior entropy because BS does not habituate) and one minor data-interpretation error (Mancini reversal asymmetry specifics) during a rigorous verification pass.

Scientific claims, model design choices, and interpretation are the author's. Claude is not an author. The complete development trajectory is recorded in the git history.

## License

MIT for code. Manuscript text under CC-BY-4.0 (eLife / bioRxiv standard).
