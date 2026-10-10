"""Dataset inspection and depth extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .validation import select_layer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Named datasets for the agent (alias -> file)
DATASETS: dict[str, Path] = {
    "EN4": DATA_DIR / "en4_sample.nc",
    "ORAS5": DATA_DIR / "oras5_sample.nc",
    "CMIP6": DATA_DIR / "cmip6_mme_sample.nc",
    "CMIP6_MME": DATA_DIR / "cmip6_mme_sample.nc",
}

ALLOWED_DEPTHS = [5.0, 100.0, 500.0, 1000.0, 2000.0]


def resolve_path(dataset_or_path: str) -> Path:
    """Resolve a dataset alias (EN4/ORAS5/CMIP6) or a file path."""
    key = dataset_or_path.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "EN4": "EN4",
        "ORAS5": "ORAS5",
        "CMIP6": "CMIP6",
        "CMIP6_MME": "CMIP6",
        "MME": "CMIP6",
    }
    if key in aliases:
        path = DATASETS[aliases[key]]
    else:
        path = Path(dataset_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"数据集不存在: {dataset_or_path} -> {path}")
    return path


def inspect_dataset(dataset_or_path: str) -> dict[str, Any]:
    """查看 NetCDF 变量、维度、深度范围等元信息。"""
    path = resolve_path(dataset_or_path)
    with xr.open_dataset(path) as ds:
        depth = ds["depth"].values if "depth" in ds.coords else None
        info = {
            "file": str(path.name),
            "variables": list(ds.data_vars),
            "dimensions": {k: int(v) for k, v in ds.sizes.items()},
            "coordinates": list(ds.coords),
            "attrs": {k: str(v) for k, v in dict(ds.attrs).items()},
        }
        if depth is not None:
            info["depth_min_m"] = float(np.min(depth))
            info["depth_max_m"] = float(np.max(depth))
            info["depths_m"] = [float(x) for x in depth]
        return info


def extract_depth(
    dataset_or_path: str,
    variable: str = "ssp",
    depth: float = 1000.0,
) -> dict[str, Any]:
    """提取指定深度的二维场摘要（不返回整场数组，避免日志过大）。"""
    path = resolve_path(dataset_or_path)
    with xr.open_dataset(path) as ds:
        da, actual = select_layer(ds, variable, depth)
        vals = da.values
        valid = np.isfinite(vals)
        return {
            "file": path.name,
            "variable": variable,
            "requested_depth_m": float(depth),
            "actual_depth_m": actual,
            "shape": list(vals.shape),
            "valid_points": int(np.count_nonzero(valid)),
            "mean": float(np.nanmean(vals)) if np.any(valid) else None,
            "min": float(np.nanmin(vals)) if np.any(valid) else None,
            "max": float(np.nanmax(vals)) if np.any(valid) else None,
        }
