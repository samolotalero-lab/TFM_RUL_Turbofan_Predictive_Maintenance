from src.data.load_data import (
    load_dataset,
    load_rul,
)

from src.data.explore_data import (
    dataset_shape,
    dataset_overview,
    dataset_quality,
)

from src.features.build_rul import build_rul
from src.features.build_test_rul import build_test_rul

from src.visualization.visualize_rul import analyze_rul
from src.visualization.visualize_correlation import sensor_correlation

from src.features.feature_selection import (
    identify_constant_sensors,
    remove_selected_sensors,
)

from src.preprocessing.preprocess import preprocess_data

# ==========================================================
# Modelos
# ==========================================================

from src.models.random_forest import (
    train_random_forest,
)

from src.models.xgboost_model import (
    train_xgboost,
)

from src.models.lstm_model import (
    train_lstm,
)

# ==========================================================
# Comparación de modelos
# ==========================================================

from src.comparison.compare_models import (
    compare_models,
)


def main():

    print("=" * 60)
    print("TFM - Remaining Useful Life Prediction")
    print("=" * 60)

    # ==========================================================
    # Carga de datos
    # ==========================================================

    train = load_dataset("data/raw/train_FD001.txt")
    test = load_dataset("data/raw/test_FD001.txt")
    rul_test = load_rul("data/raw/RUL_FD001.txt")

    print("\nProtocolo oficial NASA cargado correctamente.")

    print(f"\nMotores entrenamiento : {train['engine_id'].nunique()}")
    print(f"Motores prueba        : {test['engine_id'].nunique()}")

    # ==========================================================
    # Construcción del RUL
    # ==========================================================

    train = build_rul(train)

    test = build_test_rul(
        test,
        rul_test,
    )

    # ==========================================================
    # Exploración
    # ==========================================================

    dataset_shape(train)
    dataset_overview(train)
    dataset_quality(train)

    # ==========================================================
    # Verificación del RUL
    # ==========================================================

    print("\n" + "=" * 60)
    print("VARIABLE OBJETIVO (RUL)")
    print("=" * 60)

    print(
        train[
            [
                "engine_id",
                "cycle",
                "RUL",
            ]
        ].head(15)
    )

    print("\n" + "=" * 60)
    print("VERIFICACIÓN DEL RUL - MOTOR 1")
    print("=" * 60)

    print(
        train[
            train["engine_id"] == 1
        ][
            [
                "engine_id",
                "cycle",
                "RUL",
            ]
        ].tail(10)
    )

    # ==========================================================
    # Visualizaciones
    # ==========================================================

    analyze_rul(train)

    sensor_correlation(train)

    # ==========================================================
    # Feature Engineering
    # ==========================================================

    removed_sensors = identify_constant_sensors(
        train
    )

    train = remove_selected_sensors(
        train,
        removed_sensors,
        "train_processed.csv",
    )

    test = remove_selected_sensors(
        test,
        removed_sensors,
        "test_processed.csv",
    )

    print("\nNuevo tamaño TRAIN:")
    print(train.shape)

    print("\nNuevo tamaño TEST:")
    print(test.shape)

    # ==========================================================
    # Preprocesamiento
    # ==========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        train_info,
        test_info,
    ) = preprocess_data(
        train,
        test,
    )

    print("\n" + "=" * 60)
    print("PIPELINE PREPARADO PARA LOS MODELOS")
    print("=" * 60)

    print(f"X_train         : {X_train.shape}")
    print(f"X_test          : {X_test.shape}")
    print(f"y_train         : {y_train.shape}")
    print(f"y_test          : {y_test.shape}")

    print(f"\nX_train_scaled  : {X_train_scaled.shape}")
    print(f"X_test_scaled   : {X_test_scaled.shape}")

    print(f"\ntrain_info      : {train_info.shape}")
    print(f"test_info       : {test_info.shape}")

    # ==========================================================
    # MODELO 1 - RANDOM FOREST
    # ==========================================================

    print("\n" + "=" * 60)
    print("MODELO 1 - RANDOM FOREST")
    print("=" * 60)

    rf_model = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        test_info,
    )

    print("\nRandom Forest finalizado correctamente.")

    print("\nModelo almacenado en:")
    print("results/random_forest/")
        # ==========================================================
    # MODELO 1 - RANDOM FOREST
    # ==========================================================

    print("\n" + "=" * 60)
    print("MODELO 1 - RANDOM FOREST")
    print("=" * 60)

    rf_model = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        test_info,
    )

    print("\nRandom Forest finalizado correctamente.")

    print("\nModelo almacenado en:")
    print("results/random_forest/")

    # ==========================================================
    # MODELO 2 - XGBOOST
    # ==========================================================

    print("\n" + "=" * 60)
    print("MODELO 2 - XGBOOST")
    print("=" * 60)

    xgb_model = train_xgboost(
        X_train,
        X_test,
        y_train,
        y_test,
        test_info,
    )

    print("\nXGBoost finalizado correctamente.")

    print("\nModelo almacenado en:")
    print("results/xgboost/")

    # ==========================================================
    # MODELO 3 - LSTM
    # ==========================================================

    print("\n" + "=" * 60)
    print("MODELO 3 - LSTM")
    print("=" * 60)

    lstm_model = train_lstm(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        train_info,
        test_info,
    )

    print("\nLSTM finalizado correctamente.")

    print("\nModelo almacenado en:")
    print("results/lstm/")

    # ==========================================================
    # COMPARACIÓN DE MODELOS
    # ==========================================================

    print("\n" + "=" * 60)
    print("COMPARACIÓN AUTOMÁTICA DE MODELOS")
    print("=" * 60)

    comparison = compare_models()

    print("\nComparación finalizada correctamente.")

    print("\nResultados almacenados en:")
    print("results/comparison/")

    # ==========================================================
    # FIN DEL PIPELINE
    # ==========================================================

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO CORRECTAMENTE")
    print("=" * 60)

    print("\nEtapas completadas:")

    print("✓ Carga de datos")
    print("✓ Construcción del RUL")
    print("✓ Análisis exploratorio")
    print("✓ Ingeniería de características")
    print("✓ Preprocesamiento")
    print("✓ Random Forest")
    print("✓ XGBoost")
    print("✓ LSTM")
    print("✓ Comparación de modelos")

    print("\nResultados disponibles en:")

    print("results/random_forest/")
    print("results/xgboost/")
    print("results/lstm/")
    print("results/comparison/")

    print("\nTFM ejecutado correctamente.")


if __name__ == "__main__":
    main()
