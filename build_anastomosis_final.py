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
  %run build_anastomosis_final.py
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
print(f"\n  Coordinate units: VOXELS (not micrometres)")
print(f"  x span: {temporal['x'].max()-temporal['x'].min():.0f} voxels ≈ 50 µm embryo length")
print(f"  z span: {temporal['z'].max()-temporal['z'].min():.0f} voxels — thin confocal axis")
print(f"  eps={EPS} voxels ≈ {EPS*0.08:.1f} µm physical distance")
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

        # ── PHANTOM CHECK ───────────────────────────────────────────────────
        if set(old_m) == set(new_m):
            phantom_cnt += 1
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

anast_df = pd.DataFrame(all_events)

print(f"  Phantom events removed : {phantom_cnt}")
print(f"  Genuine events kept    : {len(anast_df)}")
print(f"  Unique switching cells : {anast_df['cell'].nunique()}")
print(f"  t_switch range         : t={int(anast_df['t_switch'].min())} to t={int(anast_df['t_switch'].max())}")


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
anast_df.to_csv(anast_path, index=False)
summary_df.to_csv(summary_path, index=False)
print(f"\n  Saved: {anast_path}  ({len(anast_df)} rows, 11 columns)")
print(f"  Saved: {summary_path}  ({len(summary_df)} rows)")


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


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Visualization: genuine events per timepoint
#
# HOW TO READ THIS CHART:
#
# X-axis: Timepoint t (developmental time; t=1 is the zygote, t=190 is
#   late embryo just before hatching).
#
# Y-axis: Number of GENUINE anastomosis events at that timepoint
#   (phantom-free — only events where the cluster membership sets changed).
#
# Bar colours:
#   Teal   — fewer than 150 genuine events (quiet period)
#   Orange — 150 to 267 events (moderate activity)
#   Red    — the single peak bar (t=115, 268 events)
#
# Shaded regions:
#   Teal region   t=53–90:  early phase. Embryo just reached the cell
#     density needed for DBSCAN to form multiple simultaneous clusters.
#     Only a handful of tight cell groups exist, so switches are rare.
#     t=53 is the absolute earliest timepoint where genuine switches
#     occur (DBSCAN produces ≥2 valid clusters at t=53 AND t=54).
#
#   Red region    t=90–150: peak phase. Corresponds to late gastrulation
#     and early organogenesis in C. elegans. Cells are actively migrating
#     to final positions (neurons to nerve ring, pharyngeal cells
#     reorganising, hypodermis migrating). Maximum cluster diversity;
#     cells cross community boundaries most frequently.
#
#   Gold region   t=150–190: late phase. Migration slows as cells settle
#     into anatomical positions. Cluster communities stabilise. Number
#     of switches declines steadily to near-zero by t=189.
#
# Dashed gold line: the peak at t=115 (268 events).
# Dotted purple line: mean events per active timepoint (~93).
#
# What this means for the DHGNN model:
#   H_anastomosis has the most columns (hyperedge columns in H_aug)
#   during t=90–150. The BCE loss receives the strongest positive
#   training signal here. The model must learn to predict community
#   switches precisely during the period of maximum morphogenetic
#   movement in the embryo.
# ═════════════════════════════════════════════════════════════════════════════
print("\nStep 7: Generating events-per-timepoint chart...")

BG = '#07070F'
AX = '#0D0D1E'

fig, ax = plt.subplots(figsize=(17, 6), facecolor=BG)
ax.set_facecolor(AX)
ax.spines[:].set_color('#2A2A3A')
ax.tick_params(colors='white', labelsize=9)

peak_t = int(summary_df.loc[summary_df['num_events'].idxmax(), 't_switch'])
peak_v = int(summary_df['num_events'].max())
mean_v = summary_df['num_events'].mean()

bar_colors = ['#FF6B6B' if t == peak_t
              else ('#FF9966' if v > 150 else '#4ECDC4')
              for t, v in zip(summary_df['t_switch'], summary_df['num_events'])]

ax.bar(summary_df['t_switch'], summary_df['num_events'],
       color=bar_colors, alpha=0.92, width=0.95)

ax.axvline(peak_t, color='#FFD93D', lw=1.8, ls='--',
           label=f'Peak:  t={peak_t}  ({peak_v} events)')
ax.axhline(mean_v, color='#A29BFE', lw=1.2, ls=':',
           label=f'Mean:  {mean_v:.0f} events / active timepoint')

ax.axvspan(53,  90,  alpha=0.07, color='#4ECDC4', label='Early  (t=53–90)')
ax.axvspan(90,  150, alpha=0.07, color='#FF6B6B', label='Peak period  (t=90–150)')
ax.axvspan(150, 190, alpha=0.07, color='#FFD93D', label='Late declining  (t=150–190)')

ax.annotate(f't={peak_t}  →  {peak_v} events',
            xy=(peak_t, peak_v), xytext=(peak_t + 7, peak_v * 0.88),
            fontsize=9, color='#FF6B6B', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.2))

ax.set_xlabel('Timepoint  t  (developmental time, t=1 = zygote, t=190 = late embryo)',
              color='white', fontsize=11)
ax.set_ylabel('Genuine anastomosis events', color='white', fontsize=11)
ax.set_title(
    f'Genuine Anastomosis Events Per Timepoint\n'
    f'{len(anast_df):,} real events  ·  {phantom_cnt} phantom events removed  ·  '
    f'DBSCAN eps={EPS} voxels (≈1.2 µm)  min_samples={MIN_SAMPLES}',
    color='white', fontsize=13, fontweight='bold')

ax.text(0.985, 0.97,
        f'Total genuine events: {len(anast_df):,}\n'
        f'Unique switching cells: {anast_df["cell"].nunique()}\n'
        f'Active timepoints: {len(summary_df)}\n'
        f'Phantoms removed: {phantom_cnt}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        color='#AAAACC',
        bbox=dict(boxstyle='round,pad=0.4', fc='#0D0D1E', alpha=0.88))

ax.legend(facecolor=AX, labelcolor='white', fontsize=8.5,
          loc='upper left', ncol=2, framealpha=0.9)
plt.tight_layout()

viz_path = os.path.join(OUTPUT_DIR, 'anastomosis_events_per_t.png')
fig.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"  Saved → {viz_path}")

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
