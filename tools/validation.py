"""Shared validation helpers for ocean NetCDF tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def ensure_variable(ds: xr.Dataset, variable: str) -> None:
    if variable not in ds.data_vars:
        available = list(ds.data_vars)
        raise ValueError(
            f"变量 '{variable}' 不存在。当前数据集可用变量: {available}。"
            f"本项目样本仅支持声速变量 ssp，不支持叶绿素等其他变量。"
        )


def ensure_depth(ds: xr.Dataset, depth: float) -> tuple[float, float]:
    """Validate depth range; return (depth_min, depth_max)."""
    if "depth" not in ds.coords:
        raise ValueError("数据缺少 depth 坐标。")
    depth_vals = ds["depth"].values
    dmin, dmax = float(np.min(depth_vals)), float(np.max(depth_vals))
    if depth < dmin - 1 or depth > dmax + 1:
        available = ", ".join(f"{float(x):.0f}" for x in depth_vals)
        raise ValueError(
            f"深度 {depth} m 越界。可用深度范围: {dmin:.0f}–{dmax:.0f} m；"
            f"样本层: {available} m。请改用可用深度后重试。"
        )
    return dmin, dmax


def select_layer(ds: xr.Dataset, variable: str, depth: float) -> tuple[Any, float]:
    ensure_variable(ds, variable)
    ensure_depth(ds, depth)
    da = ds[variable].sel(depth=depth, method="nearest")
    actual = float(da["depth"].values)
    return da, actual
