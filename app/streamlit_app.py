# app/streamlit_app.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Utilidades de carga
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def load_simulation():
    csv_path = REPORTS_DIR / "seir_simulation.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    return df


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Modelo SEIR – Chagas Valledupar",
    layout="wide",
)

st.title("Modelo SEIR aplicado a Chagas en Colombia (Valledupar)")
st.markdown(
    """
Esta aplicación permite explorar los resultados del modelo SEIR calibrado
para los casos de Chagas en Valledupar, así como el cálculo del exponente
de Lyapunov asociado a la dinámica.
"""
)

# Carga de datos
sim_df = load_simulation()
seir_params = load_json(MODELS_DIR / "seir_params.json")
lyap_result = load_json(MODELS_DIR / "lyapunov_valledupar.json")


# ---------------------------------------------------------------------
# Panel izquierdo: información general
# ---------------------------------------------------------------------
st.sidebar.header("Configuración de visualización")

if sim_df is None:
    st.sidebar.error(
        "No se encontró `reports/seir_simulation.csv`. "
        "Ejecuta primero el pipeline con `dvc repro`."
    )
else:
    # Selección de eje X
    default_x = "t" if "t" in sim_df.columns else (
        "time" if "time" in sim_df.columns else None
    )

    if default_x is None:
        x_options = ["índice (fila)"]
    else:
        x_options = [default_x]

    x_label = st.sidebar.selectbox(
        "Variable para el eje X",
        x_options,
        index=0,
    )

    # Selección de series a graficar
    numeric_cols = sim_df.select_dtypes(include=[np.number]).columns.tolist()
    if default_x in numeric_cols:
        numeric_cols.remove(default_x)

    selected_series = st.sidebar.multiselect(
        "Series a visualizar",
        options=numeric_cols,
        default=[c for c in numeric_cols if c.lower().startswith("i")][:1],
    )

# ---------------------------------------------------------------------
# Panel principal: gráficos y métricas
# ---------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Evolución temporal del modelo SEIR")

    if sim_df is None or len(sim_df) == 0:
        st.info("Aún no hay simulaciones disponibles para mostrar.")
    elif not selected_series:
        st.warning("Selecciona al menos una serie en la barra lateral.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

        if x_label == "índice (fila)":
            x = sim_df.index
            ax.set_xlabel("Paso temporal")
        else:
            x = sim_df[x_label]
            ax.set_xlabel(x_label)

        for col in selected_series:
            ax.plot(x, sim_df[col], label=col)

        ax.set_ylabel("Población / Incidencia")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Métrica simple sobre la serie seleccionada principal
        main_series = selected_series[0]
        max_val = sim_df[main_series].max()
        peak_day = sim_df[main_series].idxmax()
        st.markdown(
            f"**Pico de `{main_series}`**: {max_val:.2f} "
            f"(en el tiempo {sim_df.iloc[peak_day][x_label] if x_label != 'índice (fila)' else peak_day})"
        )

with col2:
    st.subheader("Parámetros del modelo SEIR")

    if seir_params is None:
        st.info("Aún no se encuentra `models/seir_params.json`.")
    else:
        st.json(seir_params)

    st.markdown("---")
    st.subheader("Resultado de Lyapunov")

    if lyap_result is None:
        st.info("Aún no se encuentra `models/lyapunov_valledupar.json`.")
    else:
        # Intentamos leer una clave estándar y mostramos todo por si acaso
        exponent = lyap_result.get("lyapunov_exponent") or lyap_result.get(
            "lambda_max"
        )
        if exponent is not None:
            st.metric("Exponente de Lyapunov", f"{exponent:.4f}")
        st.json(lyap_result)

st.markdown("---")
st.caption(
    "Proyecto MLOps – Modelo SEIR con cálculo de Lyapunov aplicado a casos de "
    "Chagas en Colombia (Valledupar)."
)