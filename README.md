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
│   ├── raw/                # Datos originales (controlados por .gitignore)
│   └── clean/              # Datos limpios + agregados
│
├── notebooks/
│   ├── 01_eda_chagas.ipynb                 # Exploración de datos (EDA)
│   ├── 02_feature_engineering_chagas.ipynb # Preparación de datos
│   └── 03_modelo_seir_valledupar.ipynb     # Modelo SEIR + optimización
│
├── src/
│   ├── clean_data.py
│   ├── prepare_data.py
│   ├── simulate_seir.py
│   └── calibrate_seir.py
│
├── reports/
│   └── figures/            # Gráficas finales para informes
│
├── dvc.yaml                # Pipeline de DVC
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore