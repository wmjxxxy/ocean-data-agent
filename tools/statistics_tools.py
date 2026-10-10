"""Layer statistics for ocean NetCDF fields."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .dataset_tools import resolve_path
from .validation import select_layer


def calculate_statistics(
    dataset_or_path: str,
    variable: str = "ssp",
    depth: float = 1000.0,
) -> dict[str, Any]:
    """计算指定深度层的 mean / std / min / max / valid_points。"""
    path = resolve_path(dataset_or_path)
    with xr.open_dataset(path) as ds:
        da, actual = select_layer(ds, variable, depth)
        vals = da.values.astype(float)
        valid = np.isfinite(vals)
        n = int(np.count_nonzero(valid))
        if n == 0:
            return {
                "file": path.name,
                "variable": variable,
                "actual_depth_m": actual,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "valid_points": 0,
                "message": "该深度无有效数据点",
            }

        x = vals[valid]
        return {
            "file": path.name,
            "variable": variable,
            "requested_depth_m": float(depth),
            "actual_depth_m": actual,
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "valid_points": n,
            "unit": "m/s" if variable == "ssp" else None,
        }
