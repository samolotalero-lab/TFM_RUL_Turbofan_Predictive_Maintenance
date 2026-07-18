"""
build_rul.py

Construcción de la variable Remaining Useful Life (RUL)
utilizando la estrategia Piecewise.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

import pandas as pd


RUL_CAP = 125


def build_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Remaining Useful Life (RUL) utilizando
    una estrategia Piecewise con límite superior.
    """

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .max()
        .rename("max_cycle")
    )

    df = df.merge(
        max_cycles,
        on="engine_id",
    )

    df["RUL"] = df["max_cycle"] - df["cycle"]

    # -------------------------------
    # Piecewise RUL
    # -------------------------------

    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)

    df.drop(
        columns="max_cycle",
        inplace=True,
    )

    return df

