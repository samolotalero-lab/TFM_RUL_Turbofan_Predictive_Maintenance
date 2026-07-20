"""
lstm_model.py

Modelo LSTM para la predicción del Remaining Useful Life (RUL)
utilizando el protocolo oficial NASA C-MAPSS.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
)

from tensorflow.keras.optimizers import Adam

from src.config import (
    SEQUENCE_LENGTH,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
)

from src.preprocessing.lstm_preprocessing import (
    create_sequences,
)

from src.evaluation.metrics import (
    evaluate_model,
)

from src.evaluation.plots import (
    plot_predictions,
)


def train_lstm(
    X_train,
    X_test,
    y_train,
    y_test,
    train_info,
    test_info,
):
    """
    Entrena un modelo LSTM utilizando
    el protocolo oficial NASA C-MAPSS.

    TRAIN:
        Se generan todas las ventanas posibles
        para cada motor.

    TEST:
        Únicamente se utiliza la última ventana
        temporal de cada motor.
    """

    output_path = Path("results/lstm")

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # Generación de secuencias TRAIN
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERANDO SECUENCIAS TRAIN")
    print("=" * 60)

    (
        X_train_seq,
        y_train_seq,
        train_info_seq,
    ) = create_sequences(

        X_train,

        y_train,

        train_info,

        sequence_length=SEQUENCE_LENGTH,

        last_only=False,

    )

    # ==================================================
    # Generación de secuencias TEST
    # ==================================================

    print("\n" + "=" * 60)
    print("GENERANDO SECUENCIAS TEST")
    print("=" * 60)

    (
        X_test_seq,
        y_test_seq,
        test_info_seq,
    ) = create_sequences(

        X_test,

        y_test,

        test_info,

        sequence_length=SEQUENCE_LENGTH,

        last_only=True,

    )

    # ==================================================
    # Información del dataset
    # ==================================================

    print("\n" + "=" * 60)
    print("LSTM DATASET")
    print("=" * 60)

    print(f"X_train_seq      : {X_train_seq.shape}")
    print(f"y_train_seq      : {y_train_seq.shape}")

    print()

    print(f"X_test_seq       : {X_test_seq.shape}")
    print(f"y_test_seq       : {y_test_seq.shape}")

    print()

    print(f"train_info_seq   : {train_info_seq.shape}")
    print(f"test_info_seq    : {test_info_seq.shape}")

    # ==================================================
    # Arquitectura
    # ==================================================

    model = Sequential(

        [

            Input(

                shape=(

                    X_train_seq.shape[1],

                    X_train_seq.shape[2],

                )

            ),

            LSTM(

                units=LSTM_UNITS,

                return_sequences=True,

            ),

            Dropout(

                LSTM_DROPOUT,

            ),

            LSTM(

                units=LSTM_UNITS // 2,

            ),

            Dropout(

                LSTM_DROPOUT,

            ),

            Dense(

                16,

                activation="relu",

            ),

            Dense(

                1,

            ),

        ]

    )

    # ==================================================
    # Compilación
    # ==================================================

    model.compile(

        optimizer=Adam(

            learning_rate=LSTM_LEARNING_RATE,

        ),

        loss="mse",

        metrics=[

            "mae",

        ],

    )

    # ==================================================
    # Early Stopping
    # ==================================================

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=True,

    )

    # ==================================================
    # Entrenamiento
    # ==================================================

    print("\n" + "=" * 60)
    print("ENTRENANDO LSTM")
    print("=" * 60)

    start = time.perf_counter()

    history = model.fit(

        X_train_seq,

        y_train_seq,

        validation_split=0.20,

        epochs=LSTM_EPOCHS,

        batch_size=LSTM_BATCH_SIZE,

        callbacks=[

            early_stop,

        ],

        verbose=1,

    )

    train_time = time.perf_counter() - start

    print(f"\nTiempo entrenamiento : {train_time:.2f} segundos")

    # ==================================================
    # Predicción
    # ==================================================

    print("\nRealizando predicciones...")

    start = time.perf_counter()

    predictions = model.predict(

        X_test_seq,

        verbose=0,

    ).flatten()

    prediction_time = time.perf_counter() - start

    print(f"Tiempo inferencia : {prediction_time:.4f} segundos")

    # ==================================================
    # Guardar modelo
    # ==================================================

    model.save(

        output_path / "lstm_model.keras",

    )

    # ==================================================
    # Crear DataFrame de predicciones
    # ==================================================

    prediction_df = test_info_seq.copy()

    prediction_df["Real_RUL"] = y_test_seq

    prediction_df["Predicted_RUL"] = predictions
        # ==================================================
    # Guardar predicciones
    # ==================================================

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

        model_name="lstm",

        y_true=y_test_seq,

        y_pred=predictions,

        train_time=train_time,

        prediction_time=prediction_time,

        test_info=test_info_seq,

    )

    # ==================================================
    # Predicción vs Real
    # ==================================================

    plot_predictions(

        y_test_seq,

        predictions,

        "lstm",

    )

    # ==================================================
    # Historial de entrenamiento
    # ==================================================

    history_df = pd.DataFrame(

        history.history

    )

    history_df.to_excel(

        output_path / "training_history.xlsx",

        index=False,

    )

    # ==================================================
    # Curvas de entrenamiento
    # ==================================================

    plt.figure(

        figsize=(8, 5)

    )

    plt.plot(

        history.history["loss"],

        label="Entrenamiento",

    )

    plt.plot(

        history.history["val_loss"],

        label="Validación",

    )

    plt.xlabel("Época")

    plt.ylabel("Loss")

    plt.title("Historial de entrenamiento - LSTM")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        output_path / "training_history.png",

        dpi=300,

    )

    plt.close()

    # ==================================================
    # Resumen final
    # ==================================================

    print("\n" + "=" * 60)
    print("LSTM FINALIZADO")
    print("=" * 60)

    print(f"\nModelo guardado en:")
    print(output_path / "lstm_model.keras")

    print(f"\nPredicciones guardadas en:")
    print(output_path / "predictions.csv")

    print(f"\nErrores guardados en:")
    print(output_path / "prediction_errors.csv")

    print(f"\nHistorial guardado en:")
    print(output_path / "training_history.xlsx")

    print(f"\nGráfico del entrenamiento:")
    print(output_path / "training_history.png")

    print(f"\nNúmero de motores evaluados : {len(test_info_seq)}")

    print(f"\nTiempo entrenamiento : {train_time:.2f} segundos")

    print(f"Tiempo inferencia    : {prediction_time:.4f} segundos")

    print("\nModelo LSTM entrenado correctamente.")

    return model
