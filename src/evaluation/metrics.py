"""
metrics.py

Evaluation metrics for Remaining Useful Life prediction.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==========================================================
# NASA Scoring Function
# ==========================================================

def nasa_score(
    y_true,
    y_pred,
):
    """
    NASA Scoring Function utilizada en C-MAPSS.

    Penaliza más las predicciones optimistas
    (cuando se sobreestima la vida útil restante).
    """

    score = 0.0

    for real, pred in zip(y_true, y_pred):

        d = pred - real

        if d < 0:

            score += np.exp(-d / 13.0) - 1

        else:

            score += np.exp(d / 10.0) - 1

    return score


# ==========================================================
# Evaluación completa
# ==========================================================

def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    train_time: float,
    prediction_time: float,
):
    """
    Calcula todas las métricas utilizadas en el TFM
    y guarda automáticamente todos los resultados.
    """

    results_folder = Path("results") / model_name

    results_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Métricas clásicas
    # ------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------
    # NASA Score
    # ------------------------------------------------------

    score = nasa_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------
    # Tabla de métricas
    # ------------------------------------------------------

    metrics = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
                "R2",
                "NASA Score",
                "Training Time (s)",
                "Prediction Time (s)",
            ],
            "Value": [
                mae,
                rmse,
                r2,
                score,
                train_time,
                prediction_time,
            ],
        }
    )

    metrics.to_excel(
        results_folder / "metrics.xlsx",
        index=False,
    )

    metrics.to_csv(
        results_folder / "metrics.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Error por muestra
    # ------------------------------------------------------

    prediction_errors = pd.DataFrame(
        {
            "Real_RUL": y_true,
            "Predicted_RUL": y_pred,
            "Absolute_Error": np.abs(y_true - y_pred),
            "Signed_Error": y_pred - y_true,
        }
    )

    prediction_errors.to_csv(
        results_folder / "prediction_errors.csv",
        index=False,
    )

    # ------------------------------------------------------
    # Consola
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"RESULTADOS - {model_name.upper()}")
    print("=" * 60)

    print(metrics)

    return metrics
