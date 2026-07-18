"""
feature_selection.py

Feature Engineering for NASA C-MAPSS.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path

import pandas as pd


def identify_constant_sensors(train_df: pd.DataFrame):
    """
    Identifica los sensores con varianza exactamente igual a cero
    utilizando únicamente el conjunto de entrenamiento.
    """

    tables_path = Path("results/tables")
    tables_path.mkdir(parents=True, exist_ok=True)

    sensor_columns = [
        column
        for column in train_df.columns
        if column.startswith("sensor_")
    ]

    removed = []
    kept = []

    for sensor in sensor_columns:

        variance = train_df[sensor].var()

        if variance == 0:

            removed.append([sensor, variance])

        else:

            kept.append(sensor)

    removed_df = pd.DataFrame(
        removed,
        columns=[
            "Sensor",
            "Variance",
        ],
    )

    removed_df.to_excel(
        tables_path / "removed_sensors.xlsx",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Sensores originales : {len(sensor_columns)}")
    print(f"Sensores eliminados : {len(removed)}")
    print(f"Sensores restantes  : {len(kept)}")

    if not removed_df.empty:

        print("\nSensores eliminados:\n")
        print(removed_df)

    print("\n" + "=" * 60)
    print("VARIANZA DE LOS SENSORES")
    print("=" * 60)

    print(
        train_df[sensor_columns]
        .var()
        .sort_values()
    )

    return [sensor for sensor, _ in removed]


def remove_selected_sensors(
    df: pd.DataFrame,
    removed_sensors: list,
    output_filename: str,
):
    """
    Elimina del dataset exactamente los sensores
    identificados en TRAIN y guarda el resultado.
    """

    processed_path = Path("data/processed")
    processed_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_df = df.drop(
        columns=removed_sensors
    )

    clean_df.to_csv(
        processed_path / output_filename,
        index=False,
    )

    print("\nDataset procesado guardado en:")

    print(processed_path / output_filename)

    return clean_df

