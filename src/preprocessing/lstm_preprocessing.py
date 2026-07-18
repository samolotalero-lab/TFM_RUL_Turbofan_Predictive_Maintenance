"""
lstm_preprocessing.py

Sequence generation for LSTM and GRU models.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

import numpy as np


def create_sequences(
    X,
    y,
    sequence_length=20,
):
    """
    Convierte un conjunto de datos tabular en secuencias
    para redes neuronales recurrentes (LSTM y GRU).

    Parameters
    ----------
    X : DataFrame o ndarray
        Variables predictoras.

    y : Series o ndarray
        Variable objetivo.

    sequence_length : int
        Longitud de la ventana temporal.

    Returns
    -------
    X_seq
    y_seq
    """

    X_values = np.asarray(X)
    y_values = np.asarray(y)

    X_seq = []
    y_seq = []

    for i in range(
        len(X_values) - sequence_length
    ):

        X_seq.append(
            X_values[
                i : i + sequence_length
            ]
        )

        y_seq.append(
            y_values[
                i + sequence_length
            ]
        )

    X_seq = np.array(X_seq)

    y_seq = np.array(y_seq)

    print("\n" + "=" * 60)
    print("SEQUENCE GENERATION")
    print("=" * 60)

    print(f"Sequence length : {sequence_length}")

    print(f"Input shape     : {X_values.shape}")

    print(f"Output shape    : {X_seq.shape}")

    return (
        X_seq,
        y_seq,
    )
