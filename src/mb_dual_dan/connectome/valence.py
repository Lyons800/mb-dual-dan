"""Valence assignments for larval Drosophila MB cells.

Sources:
    Saumweber et al. 2018, Nat Commun       (per-DAN optogenetic valence)
    Eichler et al. 2017, Nature             (L1 connectome + compartment naming)
    Weiglein et al. 2024, eLife 91387       (4 DL1 DANs = aversive taste punishment)
    Hu et al. 2025, eLife 100890            (DAN-c1 gustatory aversive via D2R)
    Eschbach et al. 2021, eLife 62567       (MBON output valence via chemotaxis)

Anatomical clusters:
    DL1   — aversive, vertical lobe + peduncle: DAN-c1, d1, f1, g1
    pPAM  — appetitive, medial lobe:            DAN-i1, j1, k1   (h1 absent in L1 Winding volume)

Canonical sign-inversion at MBON readout:
    MBONs in appetitive (pPAM-targeted) compartments drive AVOIDANCE
    MBONs in aversive  (DL1-targeted)   compartments drive APPROACH
    Reason: DAN-gated LTD depresses KC->MBON in trained compartments, so the
    'naive' (unlearned) MBON output is the opposite of the DAN's reinforcement
    valence (Hige 2015 logic, confirmed in larva by Eschbach 2021).
"""

from __future__ import annotations

# Per-DAN-type functional valence. 'weak' tags are anatomically appetitive but
# behaviourally inert under direct optogenetic activation.
DAN_VALENCE: dict[str, str] = {
    "DAN-c1": "aversive",
    "DAN-d1": "aversive",
    "DAN-f1": "aversive",
    "DAN-g1": "aversive",
    "DAN-i1": "appetitive",
    "DAN-j1": "appetitive",
    "DAN-k1": "appetitive",   # weak by Saumweber 2018; included as appetitive
    # "DAN-h1" is in pPAM (appetitive) but absent in L1 Winding volume
}

# Each DAN's primary compartment letter (matches MBON-x naming).
DAN_COMPARTMENT: dict[str, str] = {
    "DAN-c1": "c",
    "DAN-d1": "d",
    "DAN-f1": "f",
    "DAN-g1": "g",
    "DAN-i1": "i",
    "DAN-j1": "j",
    "DAN-k1": "k",
}

# Behavioural output direction of single-MBON optogenetic activation
# (Eschbach 2021 eLife 62567 chemotaxis assay).
MBON_BEHAVIOR: dict[str, str] = {
    "MBON-a1": "approach",
    "MBON-a2": "avoid",
    "MBON-b1": "approach",
    "MBON-b2": "approach",
    "MBON-c1": "neutral",
    "MBON-d1": "approach",
    "MBON-d2": "neutral",
    "MBON-d3": "neutral",
    "MBON-e1": "approach",
    "MBON-e2": "avoid",
    "MBON-g1": "approach",
    "MBON-g2": "approach",
    "MBON-h1": "avoid",
    "MBON-h2": "avoid",
    "MBON-i1": "avoid",
    "MBON-j1": "neutral",
    "MBON-k1": "avoid",
    "MBON-m1": "approach",
}


def dan_valence(subtype_label: str) -> str | None:
    """Return 'appetitive', 'aversive', or None for a DAN subtype string."""
    return DAN_VALENCE.get(subtype_label)


def mbon_compartment_letter(subtype_label: str) -> str | None:
    """Extract the compartment letter from an MBON subtype label.

    Examples:
        'MBON-i1'         -> 'i'
        'MBON-h1; MBON-h2'-> 'h'   (joint label; uses first MBON entry)
        'MBON-d3'         -> 'd'
    """
    if not subtype_label or not isinstance(subtype_label, str):
        return None
    parts = subtype_label.split(";")
    for p in parts:
        p = p.strip()
        if p.startswith("MBON-") and len(p) > 5:
            return p[5]
    return None


def mbon_behavior(subtype_label: str) -> str | None:
    """Return the chemotaxis-assay behavioural output for an MBON, or None."""
    parts = (subtype_label or "").split(";")
    for p in parts:
        b = MBON_BEHAVIOR.get(p.strip())
        if b is not None:
            return b
    return None
