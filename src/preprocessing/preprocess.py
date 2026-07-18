"""
preprocess.py

Preprocessing module for NASA C-MAPSS
using the official evaluation protocol.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Realiza el preprocesamiento de los datos utilizando
    el protocolo oficial de evaluación de NASA C-MAPSS.

    El StandardScaler se ajusta únicamente con el conjunto
    de entrenamiento y posteriormente se aplica al conjunto
    de prueba.
    """

    models_path = Path("results/models")
    models_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Variables predictoras - TRAIN
    # ==================================================

    X_train = train_df.drop(
        columns=[
            "engine_id",
            "cycle",
            "RUL",
        ]
    )

    y_train = train_df["RUL"]

    # ==================================================
    # Variables predictoras - TEST
    # ==================================================

    X_test = test_df.drop(
        columns=[
            "engine_id",
            "cycle",
            "RUL",
        ]
    )

    y_test = test_df["RUL"]

    # ==================================================
    # Verificación de columnas
    # ==================================================

    if list(X_train.columns) != list(X_test.columns):

        print("\n" + "=" * 60)
        print("ERROR EN LAS VARIABLES")
        print("=" * 60)

        print("\nColumnas TRAIN:")
        print(list(X_train.columns))

        print("\nColumnas TEST:")
        print(list(X_test.columns))

        raise ValueError(
            "Las columnas de TRAIN y TEST no coinciden."
        )

    # ==================================================
    # Escalado
    # ==================================================

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # ==================================================
    # Guardar StandardScaler
    # ==================================================

    joblib.dump(
        scaler,
        models_path / "standard_scaler.pkl",
    )

    # ==================================================
    # Información
    # ==================================================

    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    print(f"Variables utilizadas      : {X_train.shape[1]}")
    print(f"Motores entrenamiento     : {train_df['engine_id'].nunique()}")
    print(f"Motores prueba            : {test_df['engine_id'].nunique()}")

    print(f"\nMuestras entrenamiento    : {len(X_train)}")
    print(f"Muestras prueba           : {len(X_test)}")

    print("\ny_train                   :", y_train.shape)
    print("y_test                    :", y_test.shape)

    print("\nStandardScaler ajustado únicamente con TRAIN.")

    print("\nScaler guardado correctamente.")

    # ==================================================
    # Return
    # ==================================================

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
    )
    