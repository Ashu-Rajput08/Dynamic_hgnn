# DHGNN — Dynamic Hypergraph Neural Network for C. elegans Embryogenesis

**Open Source Cohort / DevoWorm Project**

---

## Project Overview

This project extends the DevoGraph framework by building a **Dynamic Hypergraph Neural Network (DHGNN)** to model C. elegans embryogenesis. It integrates four biologically distinct hyperedge types into a unified incidence matrix, enabling cell-level representation learning across developmental time.

### Four Hyperedge Types

| Type | Source | Fixed? | Biological Meaning |
|---|---|---|---|
| **Spatial** | `ce_temporal_data.csv` + DBSCAN | Rebuilt each t | Cells in physical proximity at timepoint t |
| **Lineage** | `cells_birth_and_pos.csv` | Fixed | Parent–daughter division bonds |
| **Functional** | `Connectome.csv` + `Alignment_map_csv.csv` | Fixed | Adult FFL neural circuit motifs |
| **Anastomosis** | `ce_temporal_data.csv` + DBSCAN | Rebuilt each t | Cells crossing spatial community boundaries |

---

## Repository Structure

```
DHGNN/
├── raw_dataset/                        ← raw input files (read-only)
│   ├── ce_temporal_data.csv            — 250,113 rows: cell positions across 190 frames
│   ├── cells_birth_and_pos.csv         — 642 rows: cell division records
│   ├── Connectome.csv                  — adult C. elegans synaptic connectome
│   ├── Alignment_map_csv.csv           — WormBase neuron→lineage name mapping
│   └── CE_cell_graph_data_processed.csv
│
├── output_tables/                      ← generated lookup tables (4 hyperedge sources)
│   ├── lineage_lookup_table.csv        — 642 parent/daughter-triplet hyperedges
│   ├── spatial_lookup_table.csv        — 5,336 per-timepoint DBSCAN community hyperedges
│   ├── functional_lookup_table.csv     — 4,436 FFL motif hyperedges (17 cols)
│   ├── anastomosis_lookup_table.csv    — 11,717 genuine (phantom-removed) switch events (11 cols)
│   └── anastomosis_phantom_events.csv  — 2,939 phantom events removed from the table above (same schema)
│
├── dhgnn_lib/                           ← shared package imported by the notebooks
│   ├── cell_universe.py                — fixed cell name <-> index universe (N = 1,625)
│   ├── node_features.py                — per-timepoint (N, 6) node feature matrices
│   ├── hyperedges.py                   — hyperedge builders + lookup-table writers
│   ├── incidence.py                    — HyperedgeSet / HyperedgeBatch + H_aug(t) assembly
│   ├── visualization.py                — plotting helpers used by the notebooks
│   ├── build_anastomosis_lookup_table.py         ← builds anastomosis_lookup_table.csv + anastomosis_phantom_events.csv
│   ├── build_anastomosis_final.py                ← original exploratory script (reference only)
│   └── build_functional_lookup_table_with_viz.py ← builds functional_lookup_table.csv
│
├── notebooks/
│   ├── run_anastomosis_functional.ipynb    — runs the two build scripts above
│   ├── 01_data_pipeline.ipynb              — cell universe, node features, lineage & spatial lookup tables
│   └── 02_incidence_matrix.ipynb           — assembles & visualizes H_aug(t)
│
├── Figures_and_Visualizations/          ← exported PNG figures, one folder per topic
│   ├── functional/                      — FFL motif grid (Gerstein-style)
│   ├── lineage/                         — generation distribution, division timing
│   ├── spatial/                         — cluster size distribution, clusters-over-time, DBSCAN scatter
│   ├── anastomosis/                     — switch events per timepoint
│   └── incidence/                       — edge type counts, sparsity pattern, size distribution, bipartite subgraph, evolution over time
│
└── README.md
```

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn networkx torch jupyter nbconvert ipykernel
```

**Step 2 — (Optional) Regenerate the functional & anastomosis lookup tables**

Run from the project root:
```bash
python dhgnn_lib/build_anastomosis_lookup_table.py             # -> output_tables/anastomosis_lookup_table.csv (11,717 rows)
                                                                 #    + output_tables/anastomosis_phantom_events.csv (2,939 rows)
python dhgnn_lib/build_functional_lookup_table_with_viz.py     # -> output_tables/functional_lookup_table.csv (4,436 rows)
```
or open `notebooks/run_anastomosis_functional.ipynb` and run all cells.

**Step 3 — Run the data pipeline & incidence matrix notebooks**

Open `notebooks/01_data_pipeline.ipynb` and `notebooks/02_incidence_matrix.ipynb` in Jupyter and run all cells, in order. Notebook 01 writes `output_tables/lineage_lookup_table.csv` and `output_tables/spatial_lookup_table.csv`; notebook 02 assembles `H_aug(t)`.

---

## Key Design Decisions

- **DBSCAN parameters (spatial + anastomosis)**: `eps=15 voxels, min_samples=3`
  Chosen to detect biologically meaningful spatial communities.

- **Phantom filter**: Switch events where `set(old_members) == set(new_members)` are removed from `anastomosis_lookup_table.csv` and kept separately in `anastomosis_phantom_events.csv`, so genuine and phantom events are never confused.

- **Duplicate handling**: `drop_duplicates('cell')` — keeps first occurrence per cell name per frame. Consistent across all timepoints.

---

## Incidence Matrix

`H_aug(t) = [ H_spatial(t) | H_lineage | H_functional | H_anastomosis(t) ]`

- **Rows**: 1,625 cells (fixed alphabetical universe)
- **Columns**: ~5,000–5,500 depending on t (642 lineage + 4,436 functional + spatial/anastomosis edges at t)
- **Storage**: sparse PyTorch COO-style tensors (cell_index, edge_index, edge_type, edge_weight) → tens of KB per timepoint

---

## Status

- [x] Phase 1 — Lookup tables (anastomosis + functional)
- [x] Phase 2 — Spatial + lineage tables + full incidence matrix
- [ ] Phase 3 — SAGNN vertex convolution
- [ ] Phase 4 — DHGNN dynamic re-clustering
