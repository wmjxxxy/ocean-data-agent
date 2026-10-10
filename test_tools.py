"""Smoke-test all tools without LLM (Day 2 checkpoint)."""

from __future__ import annotations

import json

from tools import (
    calculate_statistics,
    compare_datasets,
    extract_depth,
    inspect_dataset,
    plot_bias_map,
    plot_map,
)


def main() -> None:
    print("=== inspect_dataset(EN4) ===")
    print(json.dumps(inspect_dataset("EN4"), indent=2, ensure_ascii=False))

    print("\n=== extract_depth(EN4, ssp, 1000) ===")
    print(json.dumps(extract_depth("EN4", "ssp", 1000), indent=2, ensure_ascii=False))

    print("\n=== calculate_statistics(EN4, ssp, 1000) ===")
    print(json.dumps(calculate_statistics("EN4", "ssp", 1000), indent=2, ensure_ascii=False))

    print("\n=== compare_datasets(CMIP6, EN4, ssp, 1000) ===")
    print(json.dumps(compare_datasets("CMIP6", "EN4", "ssp", 1000), indent=2, ensure_ascii=False))

    print("\n=== plot_map(EN4, ssp, 1000) ===")
    print(json.dumps(plot_map("EN4", "ssp", 1000), indent=2, ensure_ascii=False))

    print("\n=== plot_bias_map(CMIP6, EN4, ssp, 1000) ===")
    print(json.dumps(plot_bias_map("CMIP6", "EN4", "ssp", 1000), indent=2, ensure_ascii=False))

    print("\n=== error handling: depth 7000 ===")
    try:
        calculate_statistics("EN4", "ssp", 7000)
        raise SystemExit("ERROR: expected depth overflow")
    except ValueError as e:
        print("OK:", e)

    print("\n=== error handling: chlorophyll ===")
    try:
        calculate_statistics("EN4", "chlorophyll", 1000)
        raise SystemExit("ERROR: expected missing variable")
    except ValueError as e:
        print("OK:", e)

    print("\nAll tool smoke tests passed.")


if __name__ == "__main__":
    main()
