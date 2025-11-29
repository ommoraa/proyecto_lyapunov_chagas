## Modelo SEIR aplicado a casos de Chagas en Colombia 
Este proyecto desarrolla un flujo de trabajo reproducible para el análisis epidemiológico de los casos de Chagas reportados en Colombia durante 2024. El proceso incluye limpieza, preparación y modelamiento de los datos mediante un modelo SEIR, complementado con el cálculo del exponente de Lyapunov para evaluar la estabilidad dinámica del sistema.

El municipio seleccionado para el estudio es Valledupar, elegido tras una exploración inicial por su volumen representativo de casos y la consistencia de su serie temporal.

Los datos utilizados provienen del sistema oficial Sivigila:
https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx#

(Chagas – 2024)

## 1. Objetivo del proyecto
El propósito es ajustar un modelo SEIR que permita aproximar el comportamiento temporal de los casos observados en Valledupar y evaluar si el sistema presenta sensibilidad a condiciones iniciales mediante el cálculo del exponente de Lyapunov.

- Este trabajo se estructura siguiendo buenas prácticas de ciencia de datos y MLOps:

- Versionamiento de datos con DVC

- Registro de experimentos con MLflow

- Modelamiento reproducible en Python

- Visualización y despliegue ligero con Streamlit

- Documentación del flujo completo

## 2. Estructura del proyecto
proyecto_lyapunov_chagas/
│
├── data/
│   ├── raw/                     # Datos originales sin procesar
│   │   └── Datos_2024_205.xlsx
│   │
│   ├── clean/                   # Salidas del pipeline de limpieza y preparación
│   │   ├── chagas_clean.csv
│   │   └── chagas_prepared.csv
│   │
│   └── interim/                 # Archivos intermedios
│
├── models/
│   ├── seir_params.json         # Parámetros ajustados del modelo SEIR
│   └── lyapunov_valledupar.json # Resultado del exponente de Lyapunov
│
├── notebooks/
│   ├── 01_eda_chagas.ipynb                  # Exploración y comprensión de los datos
│   ├── 02_feature_engineering_chagas.ipynb  # Transformaciones y preparación
│   └── 03_modelo_seir_valledupar.ipynb      # Ajuste y análisis del modelo SEIR
│
├── reports/
│   └── seir_simulation.csv       # Serie simulada para evaluación del modelo
│
├── src/
│   ├── __init__.py
│   ├── clean_data.py             # Procesamiento y depuración inicial
│   ├── prepare_data.py           # Feature engineering
│   ├── calibrate_seir.py         # Ajuste del modelo SEIR + MLflow tracking
│   ├── simulate_seir.py          # Funciones del modelo SEIR
│   └── compute_lyapunov.py       # Cálculo del exponente de Lyapunov
│
├── app/
│   └── streamlit_app.py          # Visualización interactiva
│
├── dvc.yaml                       # Pipeline del proyecto
├── dvc.lock
└── requirements.txt

## 3. Instalación del entorno

## 3.1. Crear y activar entorno
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

## 3.2 Instalar dependencias

pip install -r requirements.txt

## 4. Obtención de los datos
El proyecto utiliza DVC para gestionar los datos versionados. Si es la primera vez que se clona el repositorio:

dvc pull

Esto descargará los archivos necesarios en data/raw y data/clean.

## 5. Ejecución del pipeline completo

El flujo de procesamiento está definido en dvc.yaml con las siguientes etapas:

- clean_data: limpieza inicial

- prepare_data: agregaciones y transformación

- calibrate_seir: ajuste del modelo SEIR + MLflow

- compute_lyapunov: estimación del exponente de Lyapunov

Para ejecutar todo el pipeline: dvc repro

## 6. Modelo SEIR

El modelo SEIR se ajusta mediante optimización numérica (L-BFGS-B) para estimar:

- β (tasa de transmisión)

- σ (tasa de progresión expuesto → infeccioso)

- γ (tasa de recuperación)

La calibración se realiza comparando la serie simulada de infecciosos con los casos reportados en Valledupar durante 2024.

La salida principal es: 
models/seir_params.json
reports/seir_simulation.csv

Además, el script registra: parámetros, métricas, simulación y figura del ajuste en MLflow.

## 7. Registro de experimentos con MLflow

Para iniciar la interfaz de MLflow:

- mlflow ui --backend-store-uri mlruns/
- Interfaz disponible en: http://127.0.0.1:5000

Cada ejecución de calibrate_seir.py queda almacenada como un nuevo experimento con: parámetros ajustados, RMSE, artefactos del modelo y visualizaciones.

## 8. Visualización interactiva (Streamlit)

El proyecto incluye una aplicación sencilla para visualizar: 

- Casos reales

- Serie simulada SEIR

- Parámetros ajustados

- Descripción del modelo

Para iniciarla: streamlit run app/streamlit_app.py

## 9. Cálculo del exponente de Lyapunov

El archivo compute_lyapunov.py implementa una rutina para:

- linealizar el sistema alrededor de la trayectoria simulada

- estimar la tasa de divergencia de trayectorias

- obtener un indicador de estabilidad del sistema

El resultado se guarda en: models/lyapunov_valledupar.json

## 10. Reproducibilidad

Este proyecto garantiza reproducibilidad mediante:

- Versionamiento de datos y dependencias con DVC

- Registro automático de experimentos con MLflow

- Pipeline definido en dvc.yaml

- Entorno controlado mediante requirements.txt

- Código modular en src/

- Cualquier usuario puede replicar el flujo completo ejecutando:

dvc pull
dvc repro

## 11. Referencias

- Instituto Nacional de Salud – Sivigila
https://portalsivigila.ins.gov.co

- Anderson, R. M., & May, R. M. (1991). Infectious diseases of humans: Dynamics and control. Oxford University Press.

## Nota sobre la elaboración del proyecto
Este proyecto fue elaborado por el autor y contó con asistencia de ChatGPT en la corrección, depuración, estandarización y adaptación del código Python, así como en la organización de notebooks, scripts, documentación técnica y estructuración del pipeline. La responsabilidad final sobre el diseño, decisiones metodológicas y contenido analítico es completamente del autor.