# Predicción de la Vida Útil Remanente (Remaining Useful Life - RUL) mediante Inteligencia Artificial

## Trabajo Fin de Máster (TFM)

**Máster en Inteligencia Artificial**

**Autor:** Samuel Talero

---

## Descripción

Este repositorio contiene el código fuente desarrollado para el Trabajo Fin de Máster titulado:

**Comparación de modelos de Machine Learning y Deep Learning para la predicción de la Vida Útil Remanente (Remaining Useful Life - RUL) utilizando el conjunto de datos NASA C-MAPSS.**

El proyecto implementa un pipeline completo para la predicción de la vida útil remanente de motores turbofán mediante técnicas de Inteligencia Artificial aplicadas al mantenimiento predictivo.

Como caso de estudio se utiliza el subconjunto **FD001** del dataset **NASA C-MAPSS**, siguiendo el protocolo oficial de evaluación empleado por la comunidad científica.

---



# Objetivos del proyecto

El desarrollo del proyecto persigue los siguientes objetivos:

- Construir la variable objetivo Remaining Useful Life (RUL).
- Implementar un pipeline reproducible para el preprocesamiento de datos.
- Entrenar diferentes modelos de Inteligencia Artificial.
- Comparar el rendimiento de algoritmos de Machine Learning y Deep Learning.
- Analizar la viabilidad de estas técnicas para aplicaciones de mantenimiento predictivo industrial.

---



# Dataset utilizado

Se emplea el conjunto de datos **NASA C-MAPSS FD001**, ampliamente utilizado como benchmark en problemas de Prognostics and Health Management (PHM).

Características principales:

- 100 motores para entrenamiento.
- 100 motores para prueba.
- 21 sensores industriales.
- 3 variables de operación.
- Variable objetivo RUL construida mediante estrategia Piecewise (125 ciclos).

---



# Modelos implementados

Durante el desarrollo del trabajo se implementaron y compararon tres modelos predictivos:

- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

Todos los modelos fueron entrenados utilizando el mismo proceso de preprocesamiento y evaluados siguiendo el protocolo oficial de NASA C-MAPSS.

---



# Métricas de evaluación

El rendimiento de los modelos fue evaluado mediante:

- Error Absoluto Medio (MAE)
- Raíz del Error Cuadrático Medio (RMSE)
- Coeficiente de Determinación (R²)
- NASA Scoring Function
- Tiempo de entrenamiento
- Tiempo de inferencia

---



# Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

---



# Estructura del proyecto

```text

TFM_RUL_Turbofan_Predictive_Maintenance/

├── data/

│   ├── raw/

│   └── processed/

│

├── src/

│   ├── comparison/

│   ├── data/

│   ├── evaluation/

│   ├── features/

│   ├── models/

│   ├── preprocessing/

│   └── visualization/

│

├── [main.py](http://main.py)

├── requirements.txt

├── [README.md](http://README.md)

└── .gitignore

```

---



# Resultados

Los resultados obtenidos permitieron comparar el comportamiento de los modelos Random Forest, XGBoost y LSTM para la predicción del Remaining Useful Life.

El modelo LSTM obtuvo el mejor desempeño global en términos de precisión predictiva, mientras que XGBoost destacó por su eficiencia computacional y Random Forest proporcionó una solución robusta e interpretable.

---



# Autor

**Samuel Talero**

Trabajo Fin de Máster

Máster en Inteligencia Artificial

2026