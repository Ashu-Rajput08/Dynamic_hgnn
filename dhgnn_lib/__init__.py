"""Shared library code for the DHGNN notebooks.

This submission covers Phase 1 (data pipeline) and Phase 2 (incidence
matrix) only — vertex convolution, training, and validation are not
included yet.

Notebooks under ``notebooks/`` add the project root to ``sys.path`` and do::

    from dhgnn_lib.cell_universe import CellUniverse
    from dhgnn_lib import hyperedges, incidence, node_features, visualization

Modules:
    cell_universe   - fixed cell name <-> index universe (N = 1625 cells)
    node_features   - per-timepoint (N, 6) node feature matrices
    hyperedges      - builders for spatial/lineage/functional/anastomosis
                       hyperedges + writers for the lookup tables
    incidence       - HyperedgeSet / HyperedgeBatch + H_aug(t) assembly
    visualization   - plotting helpers for notebooks 01 and 02

Build scripts (also in this package, run directly from the project root,
not imported by the notebooks above):
    build_anastomosis_lookup_table.py        - builds anastomosis_lookup_table.csv
                                                + anastomosis_phantom_events.csv
    build_functional_lookup_table_with_viz.py - builds functional_lookup_table.csv
    build_anastomosis_final.py                - original exploratory script,
                                                 kept for reference only

Keeping the heavy lifting here keeps the notebooks focused on
exploration / visualization / narrative, while the underlying logic stays
unit-testable and reusable across notebooks.
"""
