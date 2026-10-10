"""Model–observation comparison metrics (Bias / RMSE / Correlation)."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .dataset_tools import resolve_path
from .validation import select_layer


def _area_weights(lat: np.ndarray, shape2d: tuple[int, int]) -> np.ndarray:
    w = np.cos(np.deg2rad(lat)).astype(float)
    w = np.clip(w, 0.0, None)
    return np.broadcast_to(w[:, None], shape2d)


def _metrics(model: np.ndarray, obs: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Area-weighted Bias, RMSE, pattern correlation — same idea as eval_common.metrics."""
    mask = np.isfinite(model) & np.isfinite(obs) & np.isfinite(weights) & (weights > 0)
    n = int(np.count_nonzero(mask))
    if n < 10:
        return {"bias": float("nan"), "rmse": float("nan"), "correlation": float("nan"), "n": n}

    diff = model[mask] - obs[mask]
    w = weights[mask]
    wsum = float(np.sum(w))
    bias = float(np.sum(diff * w) / wsum)
    rmse = float(np.sqrt(np.sum((diff**2) * w) / wsum))

    x = model[mask]
    y = obs[mask]
    x_mean = float(np.sum(x * w) / wsum)
    y_mean = float(np.sum(y * w) / wsum)
    xc = x - x_mean
    yc = y - y_mean
    denom = float(np.sqrt(np.sum(w * xc**2) * np.sum(w * yc**2)))
    corr = float(np.sum(w * xc * yc) / denom) if denom > 0 else float("nan")
    return {"bias": bias, "rmse": rmse, "correlation": corr, "n": n}


def compare_datasets(
    model_dataset: str,
    reference_dataset: str,
    variable: str = "ssp",
    depth: float = 1000.0,
) -> dict[str, Any]:
    """比较两个数据集在指定深度的 Bias / RMSE / Correlation。"""
    model_path = resolve_path(model_dataset)
    ref_path = resolve_path(reference_dataset)

    with xr.open_dataset(model_path) as ds_m, xr.open_dataset(ref_path) as ds_r:
        m, actual = select_layer(ds_m, variable, depth)
        r, _ = select_layer(ds_r, variable, depth)
        r = r.reindex_like(m, method=None)

        model_vals = m.values.astype(float)
        ref_vals = r.values.astype(float)
        lat = m["lat"].values.astype(float)
        weights = _area_weights(lat, model_vals.shape)

        metrics = _metrics(model_vals, ref_vals, weights)
        return {
            "model": model_path.name,
            "reference": ref_path.name,
            "variable": variable,
            "requested_depth_m": float(depth),
            "actual_depth_m": actual,
            "bias": metrics["bias"],
            "rmse": metrics["rmse"],
            "correlation": metrics["correlation"],
            "valid_points": metrics["n"],
            "unit": "m/s" if variable == "ssp" else None,
            "note": "Bias/RMSE 为面积加权（cos(lat)）；correlation 为 pattern correlation",
        }
