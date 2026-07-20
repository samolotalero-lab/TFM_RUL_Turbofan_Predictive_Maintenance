"""
metrics.py

Evaluation metrics for Remaining Useful Life prediction
using the official NASA C-MAPSS evaluation protocol.

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
    NASA C-MAPSS Scoring Function.

    Penaliza más las predicciones optimistas
    (sobreestimar el RUL) que las pesimistas.
    """

    score = 0.0

    for real, pred in zip(y_true, y_pred):

        error = pred - real

        if error < 0:

            score += np.exp(-error / 13.0) - 1

        else:

            score += np.exp(error / 10.0) - 1

    return score


# ==========================================================
# Evaluación completa
# ==========================================================

def evaluate_model(
    model_name,
    y_true,
    y_pred,
    train_time,
    prediction_time,
    test_info=None,
):
    """
    Evalúa un modelo de predicción del Remaining Useful Life.

    Parameters
    ----------
    model_name : str

    y_true : array-like

    y_pred : array-like

    train_time : float

    prediction_time : float

    test_info : DataFrame, opcional

        Debe contener:

            engine_id
            cycle

        Si se proporciona se añade al archivo
        prediction_errors.csv para facilitar el
        análisis posterior.
    """

    results_folder = Path("results") / model_name

    results_folder.mkdir(

        parents=True,

        exist_ok=True,

    )

    # ======================================================
    # Conversión a numpy
    # ======================================================

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    # ======================================================
    # Verificación
    # ======================================================

    if len(y_true) != len(y_pred):

        raise ValueError(

            "Las dimensiones de y_true y y_pred no coinciden."

        )

    # ======================================================
    # Métricas
    # ======================================================

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

    score = nasa_score(

        y_true,

        y_pred,

    )

    # ======================================================
    # DataFrame métricas
    # ======================================================

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

    # ======================================================
    # Guardar métricas
    # ======================================================

    metrics.to_csv(

        results_folder / "metrics.csv",

        index=False,

    )

    metrics.to_excel(

        results_folder / "metrics.xlsx",

        index=False,

    )

    # ======================================================
    # Predicciones
    # ======================================================

    prediction_errors = pd.DataFrame(

        {

            "Real_RUL": y_true,

            "Predicted_RUL": y_pred,

            "Signed_Error": (

                y_pred - y_true

            ),

            "Absolute_Error": np.abs(

                y_pred - y_true

            ),

        }

    )

    # ======================================================
    # Añadir información del motor (opcional)
    # ======================================================

    if test_info is not None:

        prediction_errors = pd.concat(

            [

                test_info.reset_index(drop=True),

                prediction_errors,

            ],

            axis=1,

        )

    # ======================================================
    # Guardar errores
    # ======================================================

    prediction_errors.to_csv(

        results_folder / "prediction_errors.csv",

        index=False,

    )

    prediction_errors.to_excel(

        results_folder / "prediction_errors.xlsx",

        index=False,

    )

    # ======================================================
    # Consola
    # ======================================================

    print("\n" + "=" * 60)

    print(f"RESULTADOS - {model_name.upper()}")

    print("=" * 60)

    print(f"\nNúmero de observaciones evaluadas : {len(y_true)}")

    if test_info is not None:

        print(

            f"Número de motores evaluados       : {test_info['engine_id'].nunique()}"

        )

    print()

    print(metrics)

    print("\nMétricas guardadas en:")

    print(results_folder / "metrics.csv")

    print(results_folder / "metrics.xlsx")

    print("\nErrores guardados en:")

    print(results_folder / "prediction_errors.csv")

    print(results_folder / "prediction_errors.xlsx")

    return metrics
