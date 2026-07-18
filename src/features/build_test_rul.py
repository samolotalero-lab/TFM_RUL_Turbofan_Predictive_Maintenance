"""
build_test_rul.py

Construcción del Remaining Useful Life (RUL)
para el conjunto de prueba siguiendo el
protocolo oficial NASA C-MAPSS.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

import pandas as pd


def build_test_rul(
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
):
    """
    Construye el RUL real del conjunto TEST
    utilizando el archivo oficial RUL_FD001.txt.
    """

    test = test_df.copy()

    # ==================================================
    # Último ciclo observado por motor
    # ==================================================

    last_cycle = (
        test.groupby("engine_id")["cycle"]
        .max()
        .rename("last_cycle")
    )

    test = test.merge(
        last_cycle,
        on="engine_id",
    )

    # ==================================================
    # Ciclo final real
    # ==================================================

    test = test.merge(
        rul_df,
        on="engine_id",
    )

    test.rename(
        columns={
            "RUL": "final_rul"
        },
        inplace=True,
    )

    test["max_cycle"] = (
        test["last_cycle"] +
        test["final_rul"]
    )

    # ==================================================
    # RUL real
    # ==================================================

    test["RUL"] = (
        test["max_cycle"] -
        test["cycle"]
    )

    # ==================================================
    # Piecewise
    # ==================================================

    test["RUL"] = test["RUL"].clip(
        upper=125
    )

    test.drop(
        columns=[
            "last_cycle",
            "final_rul",
            "max_cycle",
        ],
        inplace=True,
    )

    print("\n" + "=" * 60)
    print("TEST RUL CONSTRUIDO")
    print("=" * 60)

    print(f"Motores : {test['engine_id'].nunique()}")

    print(f"Registros : {len(test)}")

    return test
