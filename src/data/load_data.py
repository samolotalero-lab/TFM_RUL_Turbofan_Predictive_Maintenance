"""
load_data.py

Carga de los datasets oficiales NASA C-MAPSS.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path
import pandas as pd


COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Carga un archivo train/test del dataset NASA.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró {filepath}")

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
    )

    df = df.dropna(axis=1, how="all")

    df.columns = COLUMNS

    return df


def load_rul(filepath: str) -> pd.DataFrame:
    """
    Carga el archivo RUL_FD001.txt.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró {filepath}")

    rul = pd.read_csv(
        filepath,
        header=None,
        names=["RUL"],
    )

    rul["engine_id"] = range(1, len(rul) + 1)

    return rul
