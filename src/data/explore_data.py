"""
explore_data.py

Funciones para explorar el dataset NASA C-MAPSS.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

import pandas as pd


def dataset_shape(df: pd.DataFrame) -> None:
    """
    Imprime el número de filas y columnas del dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)

    print(f"Filas    : {df.shape[0]}")
    print(f"Columnas : {df.shape[1]}")


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Muestra un resumen general del dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    engines = df["engine_id"].nunique()
    max_cycle = df["cycle"].max()

    sensors = len([c for c in df.columns if c.startswith("sensor_")])
    settings = len([c for c in df.columns if c.startswith("setting_")])

    print(f"Motores                 : {engines}")
    print(f"Ciclo máximo            : {max_cycle}")
    print(f"Variables operacionales : {settings}")
    print(f"Sensores                : {sensors}")


def dataset_quality(df: pd.DataFrame) -> None:
    """
    Evalúa la calidad del conjunto de datos.
    """

    print("\n" + "=" * 60)
    print("DATASET QUALITY")
    print("=" * 60)

    missing = df.isnull().sum().sum()
    duplicated = df.duplicated().sum()
    memory = df.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"Valores nulos        : {missing}")
    print(f"Filas duplicadas     : {duplicated}")
    print(f"Memoria utilizada    : {memory:.2f} MB")

    print("\nTipos de datos:\n")
    print(df.dtypes.value_counts())
