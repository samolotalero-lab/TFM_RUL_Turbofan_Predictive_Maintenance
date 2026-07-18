"""
compare_models.py

Automatic comparison of predictive models.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def compare_models():

    output = Path("results/comparison")
    output.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # Lectura de métricas
    # ======================================================

    rf = pd.read_excel("results/random_forest/metrics.xlsx")
    xgb = pd.read_excel("results/xgboost/metrics.xlsx")
    lstm = pd.read_excel("results/lstm/metrics.xlsx")

    rf_metrics = dict(zip(rf["Metric"], rf["Value"]))
    xgb_metrics = dict(zip(xgb["Metric"], xgb["Value"]))
    lstm_metrics = dict(zip(lstm["Metric"], lstm["Value"]))

    # ======================================================
    # Tabla comparativa
    # ======================================================

    comparison = pd.DataFrame({

        "Model":[
            "Random Forest",
            "XGBoost",
            "LSTM"
        ],

        "MAE":[
            rf_metrics["MAE"],
            xgb_metrics["MAE"],
            lstm_metrics["MAE"]
        ],

        "RMSE":[
            rf_metrics["RMSE"],
            xgb_metrics["RMSE"],
            lstm_metrics["RMSE"]
        ],

        "R2":[
            rf_metrics["R2"],
            xgb_metrics["R2"],
            lstm_metrics["R2"]
        ],

        "NASA Score":[
            rf_metrics["NASA Score"],
            xgb_metrics["NASA Score"],
            lstm_metrics["NASA Score"]
        ],

        "Training Time (s)":[
            rf_metrics["Training Time (s)"],
            xgb_metrics["Training Time (s)"],
            lstm_metrics["Training Time (s)"]
        ],

        "Prediction Time (s)":[
            rf_metrics["Prediction Time (s)"],
            xgb_metrics["Prediction Time (s)"],
            lstm_metrics["Prediction Time (s)"]
        ]

    })

    # ======================================================
    # Guardar tabla
    # ======================================================

    comparison.to_excel(
        output/"comparison_models.xlsx",
        index=False
    )

    comparison.to_csv(
        output/"comparison_models.csv",
        index=False
    )

    # ======================================================
    # Función para gráficas
    # ======================================================

    def make_plot(column, filename):

        plt.figure(figsize=(7,5))

        plt.bar(
            comparison["Model"],
            comparison[column]
        )

        plt.title(column)
        plt.ylabel(column)

        plt.tight_layout()

        plt.savefig(
            output/filename,
            dpi=300
        )

        plt.close()

    # ======================================================
    # Gráficas
    # ======================================================

    make_plot("MAE","mae_comparison.png")
    make_plot("RMSE","rmse_comparison.png")
    make_plot("R2","r2_comparison.png")
    make_plot("NASA Score","nasa_score_comparison.png")
    make_plot("Training Time (s)","training_time_comparison.png")
    make_plot("Prediction Time (s)","prediction_time_comparison.png")

    # ======================================================
    # Normalización de métricas
    # ======================================================

    normalized = comparison.copy()

    # Menor es mejor
    for col in [
        "MAE",
        "RMSE",
        "NASA Score",
        "Training Time (s)",
        "Prediction Time (s)"
    ]:

        normalized[col] = (
            comparison[col].max() - comparison[col]
        ) / (
            comparison[col].max() - comparison[col].min()
        )

    # Mayor es mejor
    normalized["R2"] = (
        comparison["R2"] - comparison["R2"].min()
    ) / (
        comparison["R2"].max() - comparison["R2"].min()
    )

    # ======================================================
    # Score Global Ponderado
    # ======================================================

    normalized["Global Score"] = (

        normalized["MAE"] * 0.30 +

        normalized["RMSE"] * 0.25 +

        normalized["R2"] * 0.25 +

        normalized["NASA Score"] * 0.15 +

        normalized["Training Time (s)"] * 0.03 +

        normalized["Prediction Time (s)"] * 0.02

    )

    ranking = normalized.sort_values(
        "Global Score",
        ascending=False
    )

    ranking.insert(
        0,
        "Position",
        range(1, len(ranking)+1)
    )

    ranking.to_excel(
        output/"overall_ranking.xlsx",
        index=False
    )

    ranking.to_csv(
        output/"overall_ranking.csv",
        index=False
    )

    # ======================================================
    # Gráfico del Score Global
    # ======================================================

    plt.figure(figsize=(7,5))

    plt.bar(
        ranking["Model"],
        ranking["Global Score"]
    )

    plt.ylabel("Global Score")

    plt.title("Overall Model Ranking")

    plt.tight_layout()

    plt.savefig(
        output/"overall_ranking.png",
        dpi=300
    )

    plt.close()

    # ======================================================
    # Consola
    # ======================================================

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    print(comparison)

    print("\n" + "="*60)
    print("OVERALL RANKING")
    print("="*60)

    print(ranking[
        [
            "Position",
            "Model",
            "Global Score"
        ]
    ])

    print("\nArchivos guardados en:")
    print(output)

    return comparison
