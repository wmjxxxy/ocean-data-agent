"""Spatial map plotting for ocean fields and bias maps."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .dataset_tools import resolve_path
from .validation import select_layer

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_map(
    dataset_or_path: str,
    variable: str = "ssp",
    depth: float = 1000.0,
    title: str | None = None,
) -> dict[str, Any]:
    """绘制指定深度的空间分布图，返回 PNG 路径。"""
    path = resolve_path(dataset_or_path)
    with xr.open_dataset(path) as ds:
        da, actual = select_layer(ds, variable, depth)
        lon = da["lon"].values
        lat = da["lat"].values
        field = da.values.astype(float)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        pcm = ax.pcolormesh(lon, lat, field, shading="auto", cmap="viridis")
        cb = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("m/s" if variable == "ssp" else variable)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title or f"{path.stem} {variable} @ {actual:.0f} m")
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        fig.tight_layout()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = FIG_DIR / f"{path.stem}_{variable}_{int(actual)}m_{stamp}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)

        return {
            "file": path.name,
            "variable": variable,
            "actual_depth_m": actual,
            "figure_path": str(out),
        }


def plot_bias_map(
    model_dataset: str,
    reference_dataset: str,
    variable: str = "ssp",
    depth: float = 1000.0,
    title: str | None = None,
) -> dict[str, Any]:
    """绘制 model − reference 的 Bias 空间分布图。"""
    model_path = resolve_path(model_dataset)
    ref_path = resolve_path(reference_dataset)

    with xr.open_dataset(model_path) as ds_m, xr.open_dataset(ref_path) as ds_r:
        m, actual = select_layer(ds_m, variable, depth)
        r, _ = select_layer(ds_r, variable, depth)
        r = r.reindex_like(m, method=None)

        bias = (m.values.astype(float) - r.values.astype(float))
        lon = m["lon"].values
        lat = m["lat"].values

        finite = bias[np.isfinite(bias)]
        if finite.size == 0:
            raise ValueError("Bias 场无有效数据点，无法绘图。")

        vmax = float(np.nanpercentile(np.abs(finite), 98))
        vmax = max(vmax, 1e-6)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        pcm = ax.pcolormesh(
            lon,
            lat,
            bias,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        cb = fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Bias (m/s)" if variable == "ssp" else f"Bias ({variable})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        default_title = (
            f"{model_path.stem} − {ref_path.stem} {variable} Bias @ {actual:.0f} m"
        )
        ax.set_title(title or default_title)
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        fig.tight_layout()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = (
            FIG_DIR
            / f"bias_{model_path.stem}_vs_{ref_path.stem}_{variable}_{int(actual)}m_{stamp}.png"
        )
        fig.savefig(out, dpi=120)
        plt.close(fig)

        return {
            "model": model_path.name,
            "reference": ref_path.name,
            "variable": variable,
            "actual_depth_m": actual,
            "bias_mean": float(np.nanmean(finite)),
            "figure_path": str(out),
        }
