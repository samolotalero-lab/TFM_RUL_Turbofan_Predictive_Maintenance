"""
visualize_correlation.py

Correlation analysis between sensors and Remaining Useful Life.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def sensor_correlation(df: pd.DataFrame) -> None:

    figures_path = Path("results/figures")
    tables_path = Path("results/tables")

    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Sensores
    # -----------------------------

    sensor_columns = [
        c
        for c in df.columns
        if c.startswith("sensor_")
    ]

    correlation = (
        df[sensor_columns + ["RUL"]]
        .corr()["RUL"]
        .drop("RUL")
        .sort_values(ascending=False)
    )

    correlation.to_excel(
        tables_path / "sensor_correlations.xlsx"
    )

    print("\n" + "=" * 60)
    print("SENSOR CORRELATION RANKING")
    print("=" * 60)

    print(correlation)

    # -----------------------------
    # Heatmap
    # -----------------------------

    plt.figure(figsize=(8, 10))

    plt.imshow(
        correlation.values.reshape(-1, 1),
        aspect="auto"
    )

    plt.yticks(
        range(len(correlation)),
        correlation.index
    )

    plt.xticks(
        [0],
        ["Correlation"]
    )

    plt.colorbar()

    plt.title("Sensor Correlation with RUL")

    plt.tight_layout()

    plt.savefig(
        figures_path / "sensor_correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print("\nHeatmap guardado correctamente.")
