"""
lstm_preprocessing.py

Sequence generation for LSTM and GRU models
using the official NASA C-MAPSS protocol.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

import numpy as np
import pandas as pd


def create_sequences(
    X,
    y,
    info,
    sequence_length=20,
    last_only=False,
):
    """
    Genera secuencias independientes para cada motor.

    Parameters
    ----------
    X : DataFrame
        Variables predictoras.

    y : Series
        Variable objetivo.

    info : DataFrame
        Debe contener:

            engine_id
            cycle

    sequence_length : int

    last_only : bool
        False -> genera todas las secuencias (TRAIN)

        True -> genera únicamente la última secuencia
                de cada motor (TEST NASA)

    Returns
    -------
    X_seq

    y_seq

    info_seq
    """

    X_seq = []
    y_seq = []
    info_seq = []

    engines = info["engine_id"].unique()

    print("\n" + "=" * 60)
    print("GENERATING LSTM SEQUENCES")
    print("=" * 60)

    print(f"Motores encontrados : {len(engines)}")

    if last_only:
        print("Modo                 : ÚLTIMA SECUENCIA POR MOTOR")
    else:
        print("Modo                 : TODAS LAS SECUENCIAS")

    # ======================================================
    # Procesamiento motor por motor
    # ======================================================

    for engine in engines:

        mask = info["engine_id"] == engine

        X_engine = X.loc[mask].to_numpy()

        y_engine = y.loc[mask].to_numpy()

        info_engine = (
            info.loc[mask]
            .reset_index(drop=True)
        )

        # Si un motor no tiene suficientes ciclos
        # simplemente se ignora.

        if len(X_engine) < sequence_length:

            continue

        # ==================================================
        # SOLO ÚLTIMA SECUENCIA (NASA TEST)
        # ==================================================

        if last_only:

            X_seq.append(
                X_engine[-sequence_length:]
            )

            y_seq.append(
                y_engine[-1]
            )

            info_seq.append(
                info_engine.iloc[-1][
                    [
                        "engine_id",
                        "cycle",
                    ]
                ]
            )

        # ==================================================
        # TODAS LAS SECUENCIAS (TRAIN)
        # ==================================================

        else:

            for i in range(
                len(X_engine) - sequence_length + 1
            ):

                X_seq.append(

                    X_engine[
                        i:i + sequence_length
                    ]

                )

                y_seq.append(

                    y_engine[
                        i + sequence_length - 1
                    ]

                )

                info_seq.append(

                    info_engine.iloc[
                        i + sequence_length - 1
                    ][
                        [
                            "engine_id",
                            "cycle",
                        ]
                    ]

                )

    # ======================================================
    # Conversión
    # ======================================================

    X_seq = np.asarray(X_seq)

    y_seq = np.asarray(y_seq)

    info_seq = (
        pd.DataFrame(info_seq)
        .reset_index(drop=True)
    )

    # ======================================================
    # Información
    # ======================================================

    print("\n" + "=" * 60)
    print("SEQUENCE GENERATION")
    print("=" * 60)

    print(f"Sequence length      : {sequence_length}")

    print(f"Número de motores    : {len(engines)}")

    print(f"Secuencias generadas : {len(X_seq)}")

    print(f"X_seq shape          : {X_seq.shape}")

    print(f"y_seq shape          : {y_seq.shape}")

    print(f"info_seq shape       : {info_seq.shape}")

    print("\nLas secuencias respetan los límites de cada motor.")

    print("No existen secuencias mezclando motores.")

    # ======================================================
    # Return
    # ======================================================

    return (

        X_seq,

        y_seq,

        info_seq,

    )
