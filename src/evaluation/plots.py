"""
plots.py

Visualization module for regression models.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(
    y_true,
    y_pred,
    model_name: str,
):
    """
    Grafica valores reales vs predicciones.
    """

    output = Path("results") / model_name

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "--",
        linewidth=2
    )

    plt.xlabel("Real RUL")

    plt.ylabel("Predicted RUL")

    plt.title(f"{model_name} - Real vs Predicted")

    plt.tight_layout()

    plt.savefig(
        output / "prediction_vs_real.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


def plot_feature_importance(
    feature_names,
    importance,
    model_name: str,
):
    """
    Guarda gráfica y tabla de importancia de variables.
    """

    output = Path("results") / model_name

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance
        }
    )

    df = df.sort_values(
        "Importance",
        ascending=False
    )

    df.to_excel(
        output / "feature_importance.xlsx",
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        df["Feature"],
        df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        output / "feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

