# DHGNN — Dynamic Hypergraph Neural Network for C. elegans Embryogenesis

**GSoC / DevoWorm Project**  


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
├── output_tables/                      ← generated lookup tables
│   ├── anastomosis_final_table.csv    — 11,739 genuine switch events (11 cols)
│   └── functional_lookup_table.csv     — 4,150 FFL motif hyperedges (8 cols)
│   └── phantom_event_table.csv     -phantom events occurence record
│   └── anastomosis_final_summary.csv     
|   
│
├── build_anastomosis_final.py          ← builds anastomosis_lookup_table.csv
├── build_functional_lookup_table_with_viz.py  ← builds functional_lookup_table.csv
├── run.ipynb                  ← to run both files
├──ffl_motif_grid.png             ← feed forward loop motif representation

```

---

## How to Run

**Step 1 — Install dependencies**
```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

**Step 2 — Generate the anastomosis lookup table**
```bash
python build_anastomosis_final.py
# Output: output_tables/anastomosis_lookup_table.csv  (11,739 rows)
```

**Step 3 — Generate the functional lookup table**
```bash
python build_functional_lookup_table_with_viz.py
# Output: output_tables/functional_lookup_table.csv  (4,150 rows)
```

Both scripts read from `raw_dataset/` and write to `output_tables/`.

---

## Key Design Decisions

- **DBSCAN parameters for anastomosis**: `eps=15 voxels, min_samples=3`  
  Chosen to detect biologically meaningful spatial communities (~1.2 µm threshold).
  
- **Phantom filter**: Switch events where `set(old_members) == set(new_members)` are removed.  
  These are DBSCAN relabelling artefacts, not real community transitions.

- **Duplicate handling (anastomosis)**: `drop_duplicates('cell')` — keeps first occurrence per cell name per frame. Consistent across all timepoints.


---

## Incidence Matrix

`H_aug(t) = [ H_spatial(t) | H_lineage | H_functional | H_anastomosis(t) ]`

- **Rows**: 1,343 cells (fixed alphabetical universe)  
- **Columns**: ~5,000–5,500 depending on t  
- **Storage**: scipy sparse COO → ~20–70 KB per timepoint

---

## Status

- [x] Phase 1 — Lookup tables (anastomosis + functional)
- [ ] Phase 2 — Spatial + lineage tables + full incidence matrix
- [ ] Phase 3 — SAGNN vertex convolution
- [ ] Phase 4 — DHGNN dynamic re-clustering
