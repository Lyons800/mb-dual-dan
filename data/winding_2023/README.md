# Winding 2023 larval connectome data

Source: https://github.com/brain-networks/larval-drosophila-connectome
Paper: Winding et al. *Science* 2023, https://www.science.org/doi/10.1126/science.add9330

Files (extracted from `Supplementary-Data-S1.zip`):
- `all-all_connectivity_matrix.csv` — 2953 × 2953 signed synapse counts
- `annotations.csv` — celltype, hemilineage, clusters; 1373 hemisphere pairs
- `aa_connectivity_matrix.csv`, `ad_*`, `da_*`, `dd_*` — split by axon/dendrite source/target
- `inputs.csv`, `outputs.csv`

Note: signs (E/I) are NOT in S1. Neurotransmitter calls live in the *Science* paper's
Supplementary Table S2 — must be merged separately.

Cell-type counts of interest (paired hemispheres → ×2 for total):
- KC: 121 pairs (~242 total)
- MBON: 24
- MBIN: 14 (includes DANs + OANs + other modulatory) — ~7-8 are DANs
- PN: 103
- LN: 56
