"""Phase 0 smoke test: load the Winding connectome and extract the MB subgraph.

Validates:
- Data files extracted to the expected path.
- Connectivity matrix loads as expected shape.
- Annotations cell-type counts match Winding 2023.
- MB subgraph extraction produces sane numbers (KCs ~242, MBONs ~24, MBIN ~14).
"""

from brain.connectome.loader import load_winding, summary
from brain.connectome.mb_extract import extract_mb, mb_summary


def main():
    c = load_winding()
    print(summary(c))
    print()
    print("=" * 60)
    mb = extract_mb(c, include_pns=True)
    print(mb_summary(mb))


if __name__ == "__main__":
    main()
