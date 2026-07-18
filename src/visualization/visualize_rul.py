"""
visualize_rul.py

Visualización y estadísticas del Remaining Useful Life (RUL).

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def analyze_rul(df: pd.DataFrame) -> None:
    """
    Genera estadísticas descriptivas y un histograma del RUL.
    """

    # -----------------------------
    # Crear carpetas si no existen
    # -----------------------------

    figures_path = Path("results/figures")
    tables_path = Path("results/tables")

    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Estadísticas
    # -----------------------------

    stats = df["RUL"].describe()

    stats.to_excel(
        tables_path / "rul_statistics.xlsx"
    )

    print("\n" + "=" * 60)
    print("RUL STATISTICS")
    print("=" * 60)

    print(stats)

    # -----------------------------
    # Histograma
    # -----------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["RUL"],
        bins=40,
        edgecolor="black"
    )

    plt.title("Distribution of Remaining Useful Life")

    plt.xlabel("Remaining Useful Life (Cycles)")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        figures_path / "rul_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    print("\nFigura guardada en:")

    print(figures_path / "rul_distribution.png")

    print("\nTabla guardada en:")

    print(tables_path / "rul_statistics.xlsx")