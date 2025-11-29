# Modelo SEIR aplicado a casos de Chagas en Colombia

Este proyecto desarrolla un análisis epidemiológico de los casos de Chagas reportados en Colombia durante 2024, implementando un flujo de trabajo reproducible para la limpieza, preparación y modelamiento de datos mediante un modelo SEIR. El objetivo principal es evaluar el comportamiento de la enfermedad y ajustar parámetros que permitan aproximar la dinámica observada en un territorio específico.

El municipio seleccionado para el estudio es **Valledupar**, escogido tras un análisis exploratorio por su volumen representativo de casos y la estabilidad de su comportamiento temporal.

Los datos que se utilizaron están disponibles en: https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx#

Chagas - 2024

---

## 1. Estructura del proyecto

proyecto_lyapunov_chagas/
│
├── data/
│   ├── raw/                     # Datos originales sin procesar
│   │   └── Datos_2024_205.xlsx
│   │
│   ├── clean/                   # Datos limpiados por el pipeline
│   │   ├── chagas_clean.csv
│   │   └─- chagas_prepared.csv
│   │
│   └── interim/                 # (Opcional) Datos intermedios si se requieren
│
├── models/
│   ├── seir_params.json         # Parámetros ajustados del modelo SEIR
│   └── lyapunov_valledupar.json # Resultado del cálculo del exponente de Lyapunov
│
├── notebooks/
│   ├── 01_eda_chagas.ipynb                  # Exploración inicial y limpieza
│   ├── 02_feature_engineering_chagas.ipynb  # Preparación y derivación de variables
│   └── 03_modelo_seir_valledupar.ipynb      # Ajuste de modelo y validación
│
├── reports/
│   └── seir_simulation.csv       # Serie simulada exportada para análisis
│
├── src/
│   ├── __init__.py
│   ├── clean_data.py             # Limpieza inicial
│   ├── prepare_data.py           # Feature engineering
│   ├── calibrate_seir.py         # Ajuste base del modelo SEIR
│   ├── simulate_seir.py          # Simulación SEIR completa
│   └── compute_lyapunov.py       # Cálculo del exponente de Lyapunov
│
├── dvc.yaml                       # Pipeline completo de DVC
├── dvc.lock