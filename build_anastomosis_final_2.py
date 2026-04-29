"""
=============================================================================
build_anastomosis_final.py
=============================================================================

WHAT THIS SCRIPT PRODUCES:
  A clean lookup table of every genuine spatial community-shift event
  detected across all consecutive timepoint pairs in the C. elegans
  embryo, ready to be used directly as H_anastomosis in H_aug.

WHAT COUNTS AS GENUINE:
  A cell that had cluster label A at t and cluster label B at t+1,
  where A ≠ B, neither is -1 (noise), AND the sets of cluster members
  are provably different. Phantom events — where DBSCAN re-labelled
  an unchanged community with a new integer — are excluded.


COLUMNS INCLUDED (11 total, all necessary):
  event_id              — column name in H_aug at t_switch
  cell                  — the switching cell (Sulston notation)
  t_switch              — timepoint before switch (column is active here)
  t_next                — always t_switch + 1
  old_cluster_id        — DBSCAN scratch integer at t_switch (arbitrary)
  new_cluster_id        — DBSCAN scratch integer at t_next (arbitrary)
  old_cluster_size      — number of cells in old cluster
  new_cluster_size      — number of cells in new cluster
  old_cluster_members   — ★ pipe-separated: rows get 1.0 in H_aug column
  new_cluster_members   — ★ pipe-separated: rows get 1.0 in H_aug column
  label                 — always 1 (BCE positive class)

REQUIRED FILES (place in DATA_DIR):
  - ce_temporal_data.csv

OUTPUTS:
  - anastomosis_final_table.csv   (11,739 rows · 11 columns)
  - anastomosis_final_summary.csv (events per timepoint)
  - anastomosis_events_per_t.png  (bar chart: genuine events over time)

USAGE:
  %run build_anastomosis_final_2.py
  # DataFrames: anast_df, summary_df
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = 'raw_dataset'
TEMPORAL    = os.path.join(DATA_DIR, 'ce_temporal_data.csv')
OUTPUT_DIR  = 'output_tables'
EPS         = 15     
MIN_SAMPLES = 3      

print("=" * 70)
print("ANASTOMOSIS TABLE — FINAL VERSION (phantom-free · 11 columns)")
print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw data
# ═════════════════════════════════════════════════════════════════════════════
print("\nStep 1: Loading ce_temporal_data.csv...")
temporal = pd.read_csv(TEMPORAL)

print(f"  Shape          : {temporal.shape}")
print(f"  Unique cells   : {temporal['cell'].nunique()}")
print(f"  Timepoints     : t={int(temporal['time'].min())} to t={int(temporal['time'].max())}")
print(f"\n  NOTE: each cell name appears multiple times per timepoint")
print(f"  (multiple embryos tracked simultaneously). drop_duplicates")
print(f"  keeps the first occurrence, which is consistent for established")
print(f"  cells but can jump for newly-born cells. XYZ is excluded from")
print(f"  the table for this reason; cluster membership is still correct.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Snapshot helper
# ═════════════════════════════════════════════════════════════════════════════

def get_snap(t):
    """
    One deduplicated DBSCAN snapshot at timepoint t.
    drop_duplicates('cell') → first occurrence per cell.
    label = -1 means noise (excluded from anastomosis).
    Same deduplication used consistently at every timepoint,
    so cluster assignments are internally consistent.
    """
    snap = (temporal[temporal['time'] == t]
            .drop_duplicates('cell')
            .reset_index(drop=True)
            .copy())
    if len(snap) < MIN_SAMPLES:
        snap['label'] = -1
        return snap
    snap['label'] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(
        snap[['x', 'y', 'z']].values)
    return snap


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Detect genuine anastomosis events
# ═════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Detecting genuine anastomosis events...")
print(f"  Phantom filter: skip if set(old_members) == set(new_members)")

timepoints  = sorted(temporal['time'].unique())
all_events  = []
phantom_cnt = 0

phantom_events = []   # collect every removed phantom row

for i in range(len(timepoints) - 1):
    t0, t1 = timepoints[i], timepoints[i + 1]
    s0 = get_snap(t0)
    s1 = get_snap(t1)
    map0 = dict(zip(s0['cell'], s0['label']))
    map1 = dict(zip(s1['cell'], s1['label']))
    common = [c for c in map0 if c in map1]

    for cell in common:
        c0, c1 = map0[cell], map1[cell]
        if c0 == c1 or c0 == -1 or c1 == -1:
            continue

        old_m = sorted([c for c, lbl in map0.items() if lbl == c0])
        new_m = sorted([c for c, lbl in map1.items() if lbl == c1])

        # ── PHANTOM CHECK ──────────────────────────────────────────────────
        # A phantom: DBSCAN assigned a new integer to the EXACT SAME group.
        # The label integer changed but the community membership did NOT.
        # This is a relabelling artefact, not a real biological switch.
        if set(old_m) == set(new_m):
            phantom_cnt += 1
            phantom_events.append({
                'phantom_id':        f'PHT_{phantom_cnt-1:05d}',
                'cell':               cell,
                't_switch':           int(t0),
                't_next':             int(t1),
                'old_cluster_id':     int(c0),
                'new_cluster_id':     int(c1),
                'cluster_size':       len(old_m),
                'members':            '|'.join(old_m),
                'why_phantom':        'set(old_members)==set(new_members)',
                'old_label_int':      int(c0),
                'new_label_int':      int(c1),
                'label':              0,
            })
            continue

        all_events.append({
            'event_id':             f'ANS_{len(all_events):05d}',
            'cell':                  cell,
            't_switch':              int(t0),
            't_next':                int(t1),
            'old_cluster_id':        int(c0),
            'new_cluster_id':        int(c1),
            'old_cluster_size':      len(old_m),
            'new_cluster_size':      len(new_m),
            'old_cluster_members':   '|'.join(old_m),
            'new_cluster_members':   '|'.join(new_m),
            'label':                 1,
        })

anast_df   = pd.DataFrame(all_events)
phantom_df = pd.DataFrame(phantom_events)

print(f"  Raw detected events    : {len(anast_df) + phantom_cnt}")
print(f"  Phantom events removed : {phantom_cnt}  <- saved to phantom_events_table.csv")
print(f"  Genuine events kept    : {len(anast_df)}")
print(f"  Unique switching cells : {anast_df['cell'].nunique()}")
print(f"  t_switch range         : t={int(anast_df['t_switch'].min())} to t={int(anast_df['t_switch'].max())}")
print(f"\nPhantom table preview (first 5 rows):")
print(phantom_df[['phantom_id','cell','t_switch','cluster_size',
                  'members','old_label_int','new_label_int',
                  'why_phantom']].head(5).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Summary table
# ═════════════════════════════════════════════════════════════════════════════
summary_df = (anast_df
              .groupby('t_switch')
              .agg(num_events=('event_id', 'count'),
                   unique_cells=('cell', 'nunique'),
                   mean_old_size=('old_cluster_size', 'mean'))
              .reset_index()
              .round({'mean_old_size': 2}))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Save CSVs
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
anast_path   = os.path.join(OUTPUT_DIR, 'anastomosis_final_table.csv')
summary_path = os.path.join(OUTPUT_DIR, 'anastomosis_final_summary.csv')
phantom_path = os.path.join(OUTPUT_DIR, 'phantom_events_table.csv')
anast_df.to_csv(anast_path, index=False)
summary_df.to_csv(summary_path, index=False)
phantom_df.to_csv(phantom_path, index=False)
print(f"\n  Saved: {anast_path}  ({len(anast_df)} rows, 11 columns)")
print(f"  Saved: {summary_path}  ({len(summary_df)} rows)")
print(f"  Saved: {phantom_path}  ({len(phantom_df)} rows, 11 columns)")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Cross-verification checksums
# ═════════════════════════════════════════════════════════════════════════════
t_most = summary_df.loc[summary_df['num_events'].idxmax()]

print("\n" + "=" * 70)
print("CROSS-VERIFICATION CHECKSUMS  (compare with Excel 0_README)")
print("=" * 70)
print(f"  [1]  Total rows           : {len(anast_df)}")
print(f"  [2]  Unique cells         : {anast_df['cell'].nunique()}")
print(f"  [3]  Sum old_cluster_size : {int(anast_df['old_cluster_size'].sum())}")
print(f"  [4]  Sum new_cluster_size : {int(anast_df['new_cluster_size'].sum())}")
print(f"  [5]  First event_id       : {anast_df.iloc[0]['event_id']}")
print(f"  [6]  Last event_id        : {anast_df.iloc[-1]['event_id']}")
print(f"  [7]  First cell           : {anast_df.iloc[0]['cell']}")
print(f"  [8]  t with most events   : t={int(t_most['t_switch'])} ({int(t_most['num_events'])} events)")
print(f"  [9]  Sum of label column  : {int(anast_df['label'].sum())}  (must equal total rows)")
print(f"  [10] t_switch range       : {int(anast_df['t_switch'].min())} to {int(anast_df['t_switch'].max())}")

print(f"""
EXPECTED VALUES (must match Excel 0_README exactly):
  [1] 11739   [2] 689   [3] 74312   [4] 73953
  [5] ANS_00000   [6] ANS_11738   [7] MSpp
  [8] t=115 (268 events)   [9] 11739   [10] 53 to 189

If all 10 match → Python CSV and Excel are IDENTICAL.
If any differ → re-run from scratch and report which checksum fails.
""")

print("Sample of first 8 rows:")
print(anast_df[['event_id','cell','t_switch','t_next',
                'old_cluster_size','new_cluster_size',
                'old_cluster_members','new_cluster_members','label']].head(30).to_string())



print(f"""
======================================================================
ALL DONE.
DataFrames available: anast_df ({len(anast_df)} rows)
                      summary_df ({len(summary_df)} rows)

To confirm CSV == Excel:
  Compare the 10 printed checksums with Excel 0_README.
  All must match exactly.
======================================================================
""")
