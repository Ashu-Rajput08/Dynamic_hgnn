"""
=============================================================================
build_anastomosis_lookup_table.py
=============================================================================

WHAT THIS SCRIPT PRODUCES:
  The two anastomosis output tables actually consumed by the DHGNN pipeline
  (dhgnn_lib.cell_universe, dhgnn_lib.hyperedges):

    - output_tables/anastomosis_lookup_table.csv   (genuine, phantom-removed
      community-switch events — this IS the pipeline's H_anastomosis source)
    - output_tables/anastomosis_phantom_events.csv (the phantom events that
      were filtered out, kept separately so they are never confused with
      genuine events; same 11-column schema, event_id prefixed "PHA_")

DETECTION LOGIC (identical to build_anastomosis_final.py):
  A cell that had cluster label A at t and cluster label B at t+1, where
  A != B and neither is -1 (noise), is a candidate switch event. If the set
  of cluster members is unchanged between A and B (DBSCAN simply re-labelled
  the same community with a new integer), the event is a PHANTOM and is
  written to anastomosis_phantom_events.csv instead of the genuine table.

REQUIRED FILES (relative to project root):
  - raw_dataset/ce_temporal_data.csv

USAGE (run from the project root):
  python dhgnn_lib/build_anastomosis_lookup_table.py
=============================================================================
"""

import os
import warnings

import pandas as pd
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

DATA_DIR = "raw_dataset"
TEMPORAL = os.path.join(DATA_DIR, "ce_temporal_data.csv")
OUTPUT_DIR = "output_tables"
FIGURES_DIR = os.path.join("Figures_and_Visualizations", "anastomosis")
EPS = 15
MIN_SAMPLES = 3

COLUMNS = [
    "event_id", "cell", "t_switch", "t_next", "old_cluster_id", "new_cluster_id",
    "old_cluster_size", "new_cluster_size", "old_cluster_members", "new_cluster_members", "label",
]


def _snapshot(temporal: pd.DataFrame, t: int) -> pd.DataFrame:
    """Deduplicated DBSCAN snapshot at timepoint t (first occurrence per cell)."""
    snap = temporal[temporal["time"] == t].drop_duplicates("cell").reset_index(drop=True).copy()
    if len(snap) < MIN_SAMPLES:
        snap["label"] = -1
        return snap
    snap["label"] = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(snap[["x", "y", "z"]].values)
    return snap


def build_anastomosis_tables(raw_dir: str = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (genuine_df, phantom_df), both with the 11-column COLUMNS schema."""
    temporal = pd.read_csv(os.path.join(raw_dir, "ce_temporal_data.csv"))
    timepoints = sorted(temporal["time"].unique())

    genuine_events, phantom_events = [], []

    for i in range(len(timepoints) - 1):
        t0, t1 = timepoints[i], timepoints[i + 1]
        s0, s1 = _snapshot(temporal, t0), _snapshot(temporal, t1)
        map0 = dict(zip(s0["cell"], s0["label"]))
        map1 = dict(zip(s1["cell"], s1["label"]))

        for cell in (c for c in map0 if c in map1):
            c0, c1 = map0[cell], map1[cell]
            if c0 == c1 or c0 == -1 or c1 == -1:
                continue

            old_m = sorted(c for c, lbl in map0.items() if lbl == c0)
            new_m = sorted(c for c, lbl in map1.items() if lbl == c1)

            record = {
                "event_id": None,
                "cell": cell,
                "t_switch": int(t0),
                "t_next": int(t1),
                "old_cluster_id": int(c0),
                "new_cluster_id": int(c1),
                "old_cluster_size": len(old_m),
                "new_cluster_size": len(new_m),
                "old_cluster_members": "|".join(old_m),
                "new_cluster_members": "|".join(new_m),
                "label": 1,
            }
            (phantom_events if set(old_m) == set(new_m) else genuine_events).append(record)

    for i, r in enumerate(genuine_events):
        r["event_id"] = f"ANS_{i:05d}"
    for i, r in enumerate(phantom_events):
        r["event_id"] = f"PHA_{i:05d}"

    genuine_df = pd.DataFrame(genuine_events)[COLUMNS]
    phantom_df = pd.DataFrame(phantom_events)[COLUMNS]
    return genuine_df, phantom_df


def plot_genuine_events_per_t(genuine_df: pd.DataFrame, figures_dir: str = FIGURES_DIR) -> str:
    """Bar chart of genuine anastomosis_lookup_table.csv events per t_switch.

    Saves to figures_dir/anastomosis_genuine_events_per_t.png and returns the path.
    """
    import matplotlib.pyplot as plt

    counts = genuine_df.groupby("t_switch").size()

    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax, color="#4C72B0", width=0.9)
    ax.set_xlabel("t_switch (timepoint before the switch)")
    ax.set_ylabel("# genuine anastomosis events")
    ax.set_title("Genuine community-switch events per timepoint (anastomosis_lookup_table.csv)")
    ax.set_xticks(ax.get_xticks()[::5])
    fig.tight_layout()

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "anastomosis_genuine_events_per_t.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    genuine_df, phantom_df = build_anastomosis_tables()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    genuine_df.to_csv(os.path.join(OUTPUT_DIR, "anastomosis_lookup_table.csv"), index=False)
    phantom_df.to_csv(os.path.join(OUTPUT_DIR, "anastomosis_phantom_events.csv"), index=False)

    print(f"Genuine (phantom-removed) events : {len(genuine_df):,} -> "
          f"{OUTPUT_DIR}/anastomosis_lookup_table.csv")
    print(f"Phantom events only               : {len(phantom_df):,} -> "
          f"{OUTPUT_DIR}/anastomosis_phantom_events.csv")
    print(f"Total detected (genuine + phantom): {len(genuine_df) + len(phantom_df):,}")

    fig_path = plot_genuine_events_per_t(genuine_df)
    print(f"Figure saved                      -> {fig_path}")
