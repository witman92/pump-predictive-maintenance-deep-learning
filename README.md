# Mantenimiento predictivo en bombas industriales con deep learning

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C) ![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2) ![Optuna](https://img.shields.io/badge/Optuna-HPO-6A5ACD)

Clasificación del estado operativo de una bomba industrial (`NORMAL`, `RECOVERING`, `BROKEN`) a partir de la **serie temporal multivariada de 52 sensores** (vibración, corriente, voltaje, presión, etc.). El objetivo es detectar anomalías de forma temprana para habilitar mantenimiento predictivo.

**Pregunta:** ¿se puede clasificar de forma confiable el estado de la bomba y anticipar fallas analizando la secuencia temporal de sus sensores con arquitecturas profundas?

## Enfoque

| Etapa | Detalle |
|---|---|
| Datos | [Pump Sensor Data (Kaggle)](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data): más de 220.000 registros por minuto y solo 7 eventos de falla |
| Partición | Por **ciclos de falla** en orden cronológico, sin mezclar aleatoriamente, para evitar data leakage temporal |
| Preprocesamiento | `RobustScaler` ajustado solo en entrenamiento y **ventanas deslizantes de 60 minutos** |
| Baseline | XGBoost sobre características estadísticas de cada ventana |
| Modelos | **MLP, CNN 1D y RNN (LSTM)** en PyTorch, con entrenamiento estandarizado en `LightningModule` |
| Optimización | Optuna (búsqueda bayesiana con *pruning*) |
| Seguimiento | MLflow para curvas de pérdida/F1 e hiperparámetros, y callbacks de early stopping |

## Resultados

| Modelo | F1-Score macro | Inferencia (ms/batch) |
|---|---|---|
| **MLP** | **0.4844** | 5.43 |
| RNN (LSTM) | 0.4428 | 5.08 |
| CNN 1D | 0.4370 | — |

- El **MLP** obtuvo el mejor desempeño: la combinación de umbrales físicos dentro de la ventana de una hora explica gran parte de las anomalías, sin requerir un sesgo estrictamente secuencial.
- La **RNN** es ligeramente más rápida en inferencia, una alternativa válida para telemetría de tiempo real estricto.
- El baseline XGBoost con características estáticas perdió el contexto temporal; las redes aprovechan la ventana completa.
- La extrema rareza de las fallas (7 eventos) es la principal limitación y explica los F1 macro moderados.

## Estructura

```
notebooks/
└── pump_predictive_maintenance.ipynb   # Pipeline completo: datos → baseline → DL → Optuna → MLflow → evaluación
```

## Cómo ejecutarlo

Diseñado para Google Colab con GPU. El notebook descarga el dataset automáticamente; también puedes bajarlo desde Kaggle (`sensor.csv`, ~124 MB, no incluido en el repo).

```bash
pip install torch lightning torchmetrics mlflow optuna xgboost scikit-learn pandas numpy matplotlib seaborn
```

**Stack:** PyTorch · PyTorch Lightning · torchmetrics · MLflow · Optuna · XGBoost · scikit-learn · pandas · NumPy
