import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]

# -----------------------------
# Carga de datos con cache
# -----------------------------

@st.cache_data
def load_ajuste():
    path = BASE_DIR / "reports" / "seir_valledupar.csv"
    df = pd.read_csv(path)
    # Detectar columnas
    posibles_fecha = ["Fecha", "fecha", "date"]
    fecha_col = next((c for c in posibles_fecha if c in df.columns), None)
    if fecha_col is None:
        raise ValueError(f"No se encontró columna de fecha en {df.columns.tolist()}")
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    posibles_obs = ["casos_reales", "casos_obs", "casos", "y_true"]
    obs_col = next((c for c in posibles_obs if c in df.columns), None)
    if obs_col is None:
        raise ValueError("No se encontró columna de casos observados")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    model_candidates = [c for c in numeric_cols if c != obs_col]
    if not model_candidates:
        raise ValueError("No se encontró columna de modelo/simulado")
    # Preferimos columnas que contengan “modelo”, “sim” o “I_”
    preferidas = [
        c for c in model_candidates
        if ("modelo" in c.lower()) or ("sim" in c.lower()) or ("i_" in c.lower())
    ]
    if len(preferidas) >= 1:
        model_col = preferidas[0]
    else:
        model_col = model_candidates[-1]

    return df, fecha_col, obs_col, model_col


@st.cache_data
def load_lyapunov():
    path = BASE_DIR / "models" / "lyapunov_valledupar.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Funciones auxiliares
# -----------------------------

def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


def compute_classification_metrics(y_true, y_pred, threshold):
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_bin, y_pred)
    except ValueError:
        roc_auc = np.nan

    return y_true_bin, y_pred_bin, acc, prec, rec, f1, roc_auc


# -----------------------------
# Interfaz Streamlit
# -----------------------------

st.set_page_config(
    page_title="SEIR Chagas – Valledupar",
    layout="wide",
)

st.title("Modelo SEIR para enfermedad de Chagas – Valledupar (Cesar)")
st.write(
    """
Esta aplicación resume los resultados del proyecto de modelado SEIR para la enfermedad
de Chagas en Valledupar (Cesar). Se integran:

- Serie de casos observados y simulados.
- Métricas de ajuste como modelo de regresión.
- Clasificación de semanas de alta incidencia.
- Exponente de Lyapunov (estabilidad dinámica).
"""
)

# -----------------------------
# Carga de datos
# -----------------------------

try:
    ajuste_df, fecha_col, obs_col, model_col = load_ajuste()
except Exception as e:
    st.error(f"No fue posible cargar los datos de ajuste: {e}")
    st.stop()

lyap_info = load_lyapunov()

# Sidebar
st.sidebar.header("Configuración")

percentil_default = 0.75
percentil = st.sidebar.slider(
    "Percentil para definir alta incidencia",
    min_value=0.5,
    max_value=0.95,
    value=percentil_default,
    step=0.05,
)

y_true = ajuste_df[obs_col].astype(float).values
y_pred = ajuste_df[model_col].astype(float).values
fechas = ajuste_df[fecha_col]

umbral = float(np.quantile(y_true, percentil))
st.sidebar.write(f"Umbral actual de alta incidencia: **{umbral:.2f} casos**")

# -----------------------------
# Métricas
# -----------------------------

mae, mse, rmse, r2 = compute_regression_metrics(y_true, y_pred)
(
    y_true_bin,
    y_pred_bin,
    acc,
    prec,
    rec,
    f1,
    roc_auc,
) = compute_classification_metrics(y_true, y_pred, umbral)

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R²", f"{r2:.3f}")
col4.metric("ROC–AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/D")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Accuracy", f"{acc:.3f}")
col6.metric("Precisión", f"{prec:.3f}")
col7.metric("Recall", f"{rec:.3f}")
col8.metric("F1-score", f"{f1:.3f}")

# Lyapunov
if lyap_info is not None:
    st.subheader("Exponente de Lyapunov máximo")
    st.write(
        f"""
- Municipio: **{lyap_info.get("municipio", "Valledupar")}**  
- Exponente de Lyapunov máximo: **{lyap_info.get("lyapunov_max", 0):.6f}**  
- Pasos de integración usados: **{lyap_info.get("n_steps", 0)}**
"""
    )
else:
    st.info("No se encontró el archivo de Lyapunov. Asegúrese de haber corrido el script correspondiente.")

# -----------------------------
# Gráfico de serie temporal
# -----------------------------

st.subheader("Serie temporal de casos semanales")

fig_ts, ax = plt.subplots(figsize=(9, 4))
ax.plot(fechas, y_true, marker="o", label="Casos observados")
ax.plot(fechas, y_pred, marker="x", label="Modelo SEIR")
ax.axhline(umbral, color="red", linestyle="--", label=f"Umbral alta incidencia ({percentil:.2f})")
ax.set_xlabel("Fecha")
ax.set_ylabel("Casos semanales")
ax.legend()
ax.grid(alpha=0.3)
fig_ts.tight_layout()
st.pyplot(fig_ts)

# -----------------------------
# Matriz de confusión
# -----------------------------

st.subheader("Matriz de confusión (alta incidencia vs. baja)")

cm = confusion_matrix(y_true_bin, y_pred_bin)

fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
im = ax_cm.imshow(cm, cmap="Blues")

ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["Baja", "Alta"])
ax_cm.set_yticklabels(["Baja", "Alta"])
ax_cm.set_xlabel("Predicción")
ax_cm.set_ylabel("Real")
ax_cm.set_title("Matriz de confusión")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black",
        )

fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
fig_cm.tight_layout()
st.pyplot(fig_cm)

# -----------------------------
# Curva ROC
# -----------------------------

st.subheader("Curva ROC")

fpr, tpr, _ = roc_curve(y_true_bin, y_pred)

fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Azar")
ax_roc.set_xlabel("FPR (1 - Especificidad)")
ax_roc.set_ylabel("TPR (Sensibilidad)")
ax_roc.set_title("Curva ROC – alta incidencia")
ax_roc.legend()
ax_roc.grid(alpha=0.3)
fig_roc.tight_layout()
st.pyplot(fig_roc)

# -----------------------------
# Gráfico de residuos
# -----------------------------

st.subheader("Gráfico de residuos")

residuos = y_true - y_pred

fig_res, ax_res = plt.subplots(figsize=(6, 4))
ax_res.scatter(y_pred, residuos)
ax_res.axhline(0, color="red", linestyle="--")
ax_res.set_xlabel("Casos predichos por el modelo")
ax_res.set_ylabel("Residuo (observado – predicho)")
ax_res.set_title("Residuos del ajuste SEIR")
ax_res.grid(alpha=0.3)
fig_res.tight_layout()
st.pyplot(fig_res)

st.write(
    """
La interpretación conjunta de estas métricas y gráficos permite valorar
la utilidad del modelo SEIR como herramienta de apoyo para la vigilancia
epidemiológica de la enfermedad de Chagas en Valledupar.
"""
)