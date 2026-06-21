"""Per-timepoint node feature matrices.

For every timepoint t in [1, T] (T = 190, from ``ce_temporal_data.csv``)
builds a feature matrix ``X[t]`` of shape ``(N, F)`` over the fixed cell
universe (N cells), with feature columns::

    [x, y, z, size, lineage_generation, presence]

``presence`` is 1.0 if the cell has a recorded position at time t, else 0.0
(cell not yet born / already differentiated out of the tracked set).
``x, y, z, size`` are z-score normalised over all (cell, time) pairs.
``lineage_generation`` is static and min-max normalised by the universe max.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .cell_universe import CellUniverse, lineage_generation_from_name


class NodeFeatureBuilder:
    """Builds and caches per-timepoint node feature tensors."""

    FEATURE_NAMES = ["x", "y", "z", "size", "lineage_generation", "presence"]

    def __init__(self, raw_dir: str, universe: CellUniverse):
        self.raw_dir = raw_dir
        self.universe = universe
        self.ctd = pd.read_csv(os.path.join(raw_dir, "ce_temporal_data.csv"))
        self.timepoints = sorted(self.ctd["time"].unique().tolist())

        self._pos_mean = self.ctd[["x", "y", "z", "size"]].mean().to_numpy(dtype=np.float32)
        self._pos_std = self.ctd[["x", "y", "z", "size"]].std().to_numpy(dtype=np.float32)
        self._pos_std[self._pos_std == 0] = 1.0

        gens = np.array(
            [lineage_generation_from_name(n) for n in self.universe.names], dtype=np.float32
        )
        self._lineage_gen = gens
        self._max_gen = max(float(gens.max()), 1.0)

        self._by_time: Dict[int, pd.DataFrame] = {
            t: g.set_index("cell") for t, g in self.ctd.groupby("time")
        }

    @property
    def num_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def build(self, t: int) -> torch.Tensor:
        """Return the (N, 6) feature tensor for timepoint ``t``."""
        N = len(self.universe)
        feats = np.zeros((N, self.num_features), dtype=np.float32)
        feats[:, 4] = self._lineage_gen / self._max_gen

        frame = self._by_time.get(t)
        if frame is not None:
            present = frame.index.intersection(self.universe.names)
            for cell in present:
                row = frame.loc[cell]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                idx = self.universe.get(cell)
                vals = np.array([row["x"], row["y"], row["z"], row["size"]], dtype=np.float32)
                feats[idx, 0:4] = (vals - self._pos_mean) / self._pos_std
                feats[idx, 5] = 1.0

        return torch.from_numpy(feats)

    def raw_positions(self, t: int) -> pd.DataFrame:
        """Raw (non-normalised) x,y,z,size,cell rows present at time t."""
        frame = self._by_time.get(t)
        if frame is None:
            return pd.DataFrame(columns=["cell", "x", "y", "z", "size"])
        df = frame.reset_index()
        return df[df["cell"].isin(self.universe.names)][["cell", "x", "y", "z", "size"]]
