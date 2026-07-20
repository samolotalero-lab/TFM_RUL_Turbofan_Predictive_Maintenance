"""
config.py

Global configuration file.

Author: Samuel Talero
TFM - Máster en Inteligencia Artificial
"""

# ==========================================================
# General
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# Dataset
# ==========================================================

SEQUENCE_LENGTH = 20

# ==========================================================
# Random Forest
# ==========================================================

RF_N_ESTIMATORS = 200

RF_MAX_DEPTH = None

RF_N_JOBS = -1

# ==========================================================
# XGBoost (Optimizado para NASA FD001)
# ==========================================================

# Número de árboles
XGB_N_ESTIMATORS = 500

# Árboles ligeramente más pequeños para reducir sobreajuste
XGB_MAX_DEPTH = 4

# Learning rate más conservador
XGB_LEARNING_RATE = 0.05

# Utilizar el 90% de las muestras por árbol
XGB_SUBSAMPLE = 0.90

# Utilizar el 90% de las variables por árbol
XGB_COLSAMPLE = 0.90

# Regularización L1
XGB_REG_ALPHA = 0.10

# Regularización L2
XGB_REG_LAMBDA = 1.50

# ==========================================================
# LSTM
# ==========================================================

LSTM_UNITS = 64

LSTM_DROPOUT = 0.20

LSTM_EPOCHS = 50

LSTM_BATCH_SIZE = 64

LSTM_LEARNING_RATE = 0.001

# ==========================================================
# EarlyStopping
# ==========================================================

EARLY_STOPPING_PATIENCE = 10
