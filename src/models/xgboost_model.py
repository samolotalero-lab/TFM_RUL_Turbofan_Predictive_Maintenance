"""
xgboost_model.py

XGBoost model for Remaining Useful Life prediction.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path
import time

import joblib
import pandas as pd

from xgboost import XGBRegressor

from src.config import (
    RANDOM_STATE,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE,
)

from src.evaluation.metrics import evaluate_model
from src.evaluation.plots import (
    plot_predictions,
    plot_feature_importance,
)


def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Entrena un modelo XGBoost utilizando el
    protocolo oficial de evaluación NASA C-MAPSS.
    """

    output_path = Path("results/xgboost")
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Configuración del modelo
    # ==================================================

    model = XGBRegressor(

        objective="reg:squarederror",

        n_estimators=XGB_N_ESTIMATORS,

        learning_rate=XGB_LEARNING_RATE,

        max_depth=XGB_MAX_DEPTH,

        subsample=XGB_SUBSAMPLE,

        colsample_bytree=XGB_COLSAMPLE,

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )

    print("\n" + "=" * 60)
    print("XGBOOST")
    print("=" * 60)

    print("\nEntrenando modelo...")

    # ==================================================
    # Entrenamiento
    # ==================================================

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    train_time = time.perf_counter() - start

    print(f"Tiempo entrenamiento : {train_time:.3f} segundos")

    # ==================================================
    # Predicción
    # ==================================================

    print("\nRealizando predicciones...")

    start = time.perf_counter()

    predictions = model.predict(
        X_test
    )

    prediction_time = time.perf_counter() - start

    print(f"Tiempo inferencia : {prediction_time:.4f} segundos")

    # ==================================================
    # Guardar modelo
    # ==================================================

    joblib.dump(
        model,
        output_path / "xgboost.pkl",
    )

    # ==================================================
    # Guardar predicciones
    # ==================================================

    prediction_df = pd.DataFrame(
        {
            "Real_RUL": y_test.values,
            "Predicted_RUL": predictions,
        }
    )

    prediction_df.to_csv(
        output_path / "predictions.csv",
        index=False,
    )

    # ==================================================
    # Guardar errores
    # ==================================================

    errors = prediction_df.copy()

    errors["Error"] = (
        errors["Predicted_RUL"]
        - errors["Real_RUL"]
    )

    errors["Absolute_Error"] = (
        errors["Error"].abs()
    )

    errors.to_csv(
        output_path / "prediction_errors.csv",
        index=False,
    )

    # ==================================================
    # Evaluación
    # ==================================================

    evaluate_model(
        model_name="xgboost",
        y_true=y_test,
        y_pred=predictions,
        train_time=train_time,
        prediction_time=prediction_time,
    )

    # ==================================================
    # Predicción vs Real
    # ==================================================

    plot_predictions(
        y_test,
        predictions,
        "xgboost",
    )

    # ==================================================
    # Importancia de variables
    # ==================================================

    plot_feature_importance(
        X_train.columns,
        model.feature_importances_,
        "xgboost",
    )

    importance = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    importance.to_excel(
        output_path / "feature_importance.xlsx",
        index=False,
    )

    importance.head(10).to_excel(
        output_path / "feature_importance_top10.xlsx",
        index=False,
    )

    # ==================================================
    # Consola
    # ==================================================

    print("\nTop 10 variables más importantes:\n")

    print(importance.head(10))

    print("\nModelo XGBoost entrenado correctamente.")

    print(f"\nResultados guardados en:\n{output_path}")

    return model

