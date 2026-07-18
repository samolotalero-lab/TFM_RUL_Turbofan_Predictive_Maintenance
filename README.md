# # Predicción de la Vida Útil Remanente (Remaining Useful Life - RUL) mediante Machine Learning y Deep Learning

## Trabajo Fin de Máster

**Autor:** Samuel Talero

---

## Descripción del proyecto

Este proyecto implementa un sistema de mantenimiento predictivo para estimar la Vida Útil Remanente (Remaining Useful Life - RUL) de motores aeronáuticos utilizando el conjunto de datos NASA C-MAPSS FD001.

Se desarrollaron y compararon tres modelos predictivos:

- Random Forest

- XGBoost

- Long Short-Term Memory (LSTM)

El pipeline completo implementa:

- Carga de datos

- Análisis exploratorio de datos (EDA)

- Ingeniería de características

- Preprocesamiento

- Entrenamiento de modelos

- Evaluación mediante múltiples métricas

- Comparación automática de modelos

---

## Dataset utilizado

NASA C-MAPSS FD001

---

## Métricas de evaluación

- Mean Absolute Error (MAE)

- Root Mean Squared Error (RMSE)

- Coeficiente de determinación (R²)

- NASA Scoring Function

---

## Tecnologías utilizadas

- Python

- Scikit-Learn

- TensorFlow / Keras

- XGBoost

- Pandas

- NumPy

- Matplotlib

---

## Estructura del proyecto

```

data/

results/

src/

[main.py](http://main.py)

requirements.txt

[README.md](http://README.md)

```

---

## Ejecución

```bash

python [main.py](http://main.py)

```

---

## Resultados

Todos los modelos entrenados, métricas, figuras, predicciones y comparaciones se almacenan automáticamente dentro de la carpeta:

```

results/

```