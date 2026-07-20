"""
compare_models.py

Automatic comparison of predictive models.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Función auxiliar
# ==========================================================

def normalize_metric(series, higher_is_better=False):
    """
    Normaliza una métrica entre 0 y 1.

    Si todos los valores son iguales,
    devuelve 1 para todos los modelos.
    """

    maximum = series.max()
    minimum = series.min()

    if maximum == minimum:
        return pd.Series(
            [1.0] * len(series),
            index=series.index,
        )

    if higher_is_better:

        return (

            series - minimum

        ) / (

            maximum - minimum

        )

    return (

        maximum - series

    ) / (

        maximum - minimum

    )


# ==========================================================
# Comparación
# ==========================================================

def compare_models():

    output = Path("results/comparison")

    output.mkdir(

        parents=True,

        exist_ok=True,

    )

    # ======================================================
    # Lectura de métricas
    # ======================================================

    models = {

        "Random Forest": "results/random_forest/metrics.xlsx",

        "XGBoost": "results/xgboost/metrics.xlsx",

        "LSTM": "results/lstm/metrics.xlsx",

    }

    metrics_dict = {}

    for model_name, file_path in models.items():

        metrics = pd.read_excel(file_path)

        metrics_dict[model_name] = dict(

            zip(

                metrics["Metric"],

                metrics["Value"],

            )

        )

    # ======================================================
    # Tabla comparativa
    # ======================================================

    comparison = pd.DataFrame(

        {

            "Model": list(models.keys()),

            "MAE": [

                metrics_dict[m]["MAE"]

                for m in models

            ],

            "RMSE": [

                metrics_dict[m]["RMSE"]

                for m in models

            ],

            "R2": [

                metrics_dict[m]["R2"]

                for m in models

            ],

            "NASA Score": [

                metrics_dict[m]["NASA Score"]

                for m in models

            ],

            "Training Time (s)": [

                metrics_dict[m]["Training Time (s)"]

                for m in models

            ],

            "Prediction Time (s)": [

                metrics_dict[m]["Prediction Time (s)"]

                for m in models

            ],

        }

    )

    # ======================================================
    # Guardar comparación
    # ======================================================

    comparison.to_csv(

        output / "comparison_models.csv",

        index=False,

    )

    comparison.to_excel(

        output / "comparison_models.xlsx",

        index=False,

    )

    # ======================================================
    # Función para generar gráficas
    # ======================================================

    def make_plot(column, filename):

        plt.figure(

            figsize=(7, 5)

        )

        plt.bar(

            comparison["Model"],

            comparison[column],

        )

        plt.title(column)

        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(

            output / filename,

            dpi=300,

        )

        plt.close()
            # ======================================================
    # Generación de gráficas
    # ======================================================

    make_plot(

        "MAE",

        "mae_comparison.png",

    )

    make_plot(

        "RMSE",

        "rmse_comparison.png",

    )

    make_plot(

        "R2",

        "r2_comparison.png",

    )

    make_plot(

        "NASA Score",

        "nasa_score_comparison.png",

    )

    make_plot(

        "Training Time (s)",

        "training_time_comparison.png",

    )

    make_plot(

        "Prediction Time (s)",

        "prediction_time_comparison.png",

    )

    # ======================================================
    # Normalización de métricas
    # ======================================================

    normalized = comparison.copy()

    normalized["MAE"] = normalize_metric(

        comparison["MAE"]

    )

    normalized["RMSE"] = normalize_metric(

        comparison["RMSE"]

    )

    normalized["NASA Score"] = normalize_metric(

        comparison["NASA Score"]

    )

    normalized["Training Time (s)"] = normalize_metric(

        comparison["Training Time (s)"]

    )

    normalized["Prediction Time (s)"] = normalize_metric(

        comparison["Prediction Time (s)"]

    )

    normalized["R2"] = normalize_metric(

        comparison["R2"],

        higher_is_better=True,

    )

    # ======================================================
    # Global Score
    # ======================================================

    normalized["Global Score"] = (

        normalized["MAE"] * 0.30 +

        normalized["RMSE"] * 0.25 +

        normalized["R2"] * 0.25 +

        normalized["NASA Score"] * 0.15 +

        normalized["Training Time (s)"] * 0.03 +

        normalized["Prediction Time (s)"] * 0.02

    )

    # ======================================================
    # Ranking final
    # ======================================================

    ranking = (

        normalized

        .sort_values(

            by="Global Score",

            ascending=False,

        )

        .reset_index(drop=True)

    )

    ranking.insert(

        0,

        "Position",

        range(

            1,

            len(ranking) + 1,

        ),

    )

    # ======================================================
    # Guardar ranking
    # ======================================================

    ranking.to_csv(

        output / "overall_ranking.csv",

        index=False,

    )

    ranking.to_excel(

        output / "overall_ranking.xlsx",

        index=False,

    )

    # ======================================================
    # Gráfico Global Score
    # ======================================================

    plt.figure(

        figsize=(8, 5)

    )

    plt.bar(

        ranking["Model"],

        ranking["Global Score"],

    )

    plt.ylabel(

        "Global Score"

    )

    plt.title(

        "Overall Model Ranking"

    )

    plt.tight_layout()

    plt.savefig(

        output / "overall_ranking.png",

        dpi=300,

    )

    plt.close()

    # ======================================================
    # Resumen por consola
    # ======================================================

    print("\n" + "=" * 60)

    print("MODEL COMPARISON")

    print("=" * 60)

    print()

    print(comparison)

    print("\n" + "=" * 60)

    print("GLOBAL RANKING")

    print("=" * 60)

    print()

    print(

        ranking[

            [

                "Position",

                "Model",

                "Global Score",

            ]

        ]

    )

    print("\nBest model:")

    print(

        f"   {ranking.iloc[0]['Model']}"

    )

    print(

        f"   Global Score: {ranking.iloc[0]['Global Score']:.4f}"

    )

    print("\nFiles generated:")

    print(f"   {output/'comparison_models.xlsx'}")

    print(f"   {output/'comparison_models.csv'}")

    print(f"   {output/'overall_ranking.xlsx'}")

    print(f"   {output/'overall_ranking.csv'}")

    print(f"   {output/'overall_ranking.png'}")

    print("\nComparison finished successfully.")

    return comparison
