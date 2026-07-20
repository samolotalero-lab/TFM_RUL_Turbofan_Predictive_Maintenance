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

from src.evaluation.metrics import (
    evaluate_model,
)

from src.evaluation.plots import (
    plot_predictions,
    plot_feature_importance,
)


def train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    test_info,
):
    """
    Entrena un modelo XGBoost utilizando
    el protocolo oficial NASA C-MAPSS.

    La evaluación se realiza únicamente
    sobre la última observación de cada motor.
    """

    output_path = Path("results/xgboost")

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Modelo
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

    print(f"Tiempo entrenamiento : {train_time:.2f} segundos")

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
    # Construcción del DataFrame de predicciones
    # ==================================================

    prediction_df = test_info.copy()

    prediction_df["Real_RUL"] = y_test.values

    prediction_df["Predicted_RUL"] = predictions

    # ==================================================
    # Protocolo oficial NASA
    # Mantener únicamente la última observación
    # de cada motor
    # ==================================================

    prediction_df = (

        prediction_df

        .sort_values(
            ["engine_id", "cycle"]
        )

        .groupby("engine_id")

        .tail(1)

        .reset_index(drop=True)

    )

    # ==================================================
    # Errores
    # ==================================================

    prediction_df["Signed_Error"] = (

        prediction_df["Predicted_RUL"]

        - prediction_df["Real_RUL"]

    )

    prediction_df["Absolute_Error"] = (

        prediction_df["Signed_Error"].abs()

    )

    # ==================================================
    # Guardar resultados
    # ==================================================

    prediction_df.to_csv(

        output_path / "predictions.csv",

        index=False,

    )

    prediction_df.to_csv(

        output_path / "prediction_errors.csv",

        index=False,

    )

    # ==================================================
    # Evaluación oficial NASA
    # ==================================================

    evaluate_model(

        model_name="xgboost",

        y_true=prediction_df["Real_RUL"],

        y_pred=prediction_df["Predicted_RUL"],

        train_time=train_time,

        prediction_time=prediction_time,

        test_info=None,

    )

    # ==================================================
    # Predicción vs Real
    # ==================================================

    plot_predictions(

        prediction_df["Real_RUL"],

        prediction_df["Predicted_RUL"],

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
    # Información protocolo NASA
    # ==================================================

    print("\n" + "=" * 60)
    print("PROTOCOLO OFICIAL NASA")
    print("=" * 60)

    print(
        f"Motores evaluados : {len(prediction_df)}"
    )

    print(
        f"Predicciones usadas para NASA Score : {len(prediction_df)}"
    )

    print(
        "\n(Se utiliza únicamente la última predicción de cada motor)"
    )

    # ==================================================
    # Top variables
    # ==================================================

    print("\nTop 10 variables más importantes:\n")

    print(
        importance.head(10)
    )

    # ==================================================
    # Resumen
    # ==================================================

    print("\n" + "=" * 60)
    print("XGBOOST FINALIZADO")
    print("=" * 60)

    print("\nModelo guardado en:")
    print(output_path / "xgboost.pkl")

    print("\nPredicciones guardadas en:")
    print(output_path / "predictions.csv")

    print("\nErrores guardados en:")
    print(output_path / "prediction_errors.csv")

    print("\nImportancia de variables:")
    print(output_path / "feature_importance.xlsx")

    print(output_path / "feature_importance_top10.xlsx")

    print(f"\nTiempo entrenamiento : {train_time:.2f} segundos")

    print(f"Tiempo inferencia    : {prediction_time:.4f} segundos")

    print("\nModelo XGBoost entrenado correctamente.")

    print(f"\nResultados guardados en:\n{output_path}")

    return model
