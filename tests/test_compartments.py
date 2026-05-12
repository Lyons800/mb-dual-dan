"""Phase 3b regression tests: pin down compartment-discovery numbers."""

from brain.connectome import extract_mb, load_winding
from brain.connectome.compartments import compartment_groups, discover_bipartite


def test_jaccard_matrix_shape_and_max():
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.6)
    assert bi.jaccard.shape == (28, 48)
    # the larval MB has at least one pair of (DAN, MBON) sharing >75% of its KC pool
    assert bi.jaccard.max() > 0.75


def test_compartment_count_at_tight_threshold():
    """At Jaccard >= 0.6 we expect a handful of compartment-like groups."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.6)
    groups = compartment_groups(bi)
    assert 3 <= len(groups) <= 10
    # the largest group should contain MULTIPLE MBINs (broadcast core)
    biggest = max(groups, key=lambda g: g["n_mbin"] + g["n_mbon"])
    assert biggest["n_mbin"] >= 5


def test_tight_compartments_exist():
    """At Jaccard >= 0.7 only the tightest DAN-MBON pairings survive."""
    c = load_winding()
    mb = extract_mb(c, include_pns=False)
    bi = discover_bipartite(mb, threshold=0.7)
    assert len(bi.pairings) >= 10
    # Top pair is a known-high overlap canonical compartment-like
    top = bi.pairings[0]
    assert top[2] > 0.7
