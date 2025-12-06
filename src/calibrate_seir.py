from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# MLflow opcional: si no está instalado, el script sigue funcionando
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


# Rutas base del proyecto
BASE_DIR = Path(__file__).resolve().parents[1]
TS_FOCO_PATH = BASE_DIR / "data" / "clean" / "chagas_ts_foco.csv"
PARAMS_BASE_PATH = BASE_DIR / "models" / "seir_params.json"
PARAMS_OPT_PATH = BASE_DIR / "models" / "seir_params_opt.json"
AJUSTE_CSV_PATH = BASE_DIR / "reports" / "seir_valledupar.csv"
AJUSTE_FIG_PATH = BASE_DIR / "reports" / "seir_valledupar.png"
RESID_FIG_PATH = BASE_DIR / "reports" / "residuos_seir_valledupar.png"
ROC_FIG_PATH = BASE_DIR / "reports" / "roc_seir_valledupar.png"
CM_FIG_PATH = BASE_DIR / "reports" / "cm_seir_valledupar.png"


# -------------------------------------------------------------------
# 1. Sistema SEIR continuo
# -------------------------------------------------------------------


def seir_system(t, y, p):
    """
    Sistema SEIR con incidencia no lineal y migración:
    y = [S, E, I, R, V]
    """
    S, E, I, R, V = y

    beta = p["beta"]
    alpha = p["alpha"]
    sigma = p["sigma"]
    gamma = p["gamma"]
    mu = p["mu"]
    c = p["c"]
    d = p["d"]
    m_out = p.get("m_out", 0.0)
    Lambda = p["Lambda"]

    N = S + E + I + R
    if N <= 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    infection = beta * S * V / (1.0 + alpha * I)

    dSdt = Lambda - infection - (mu + m_out) * S
    dEdt = infection - (sigma + mu + m_out) * E
    dIdt = sigma * E - (gamma + mu + m_out) * I
    dRdt = gamma * I - (mu + m_out) * R
    dVdt = c * I - d * V

    return [dSdt, dEdt, dIdt, dRdt, dVdt]


def simulate_seir(params: dict, n_days: int):
    """
    Simula el SEIR durante n_days (en días).
    Devuelve t, y donde y tiene 5 filas: S, E, I, R, V.
    """
    p = params.copy()

    N0 = float(p["N0"])
    I0 = float(p.get("I0", 1.0))
    E0 = float(p.get("E0", 0.0))
    R0 = 0.0
    V0 = 1.0
    S0 = N0 - I0 - E0 - R0

    # Asegurar Lambda = mu * N0 si no está definida
    p.setdefault("Lambda", p["mu"] * N0)

    t_span = (0.0, float(n_days))
    t_eval = np.arange(0, n_days + 1)

    sol = solve_ivp(
        lambda t, y: seir_system(t, y, p),
        t_span=t_span,
        y0=[S0, E0, I0, R0, V0],
        t_eval=t_eval,
        vectorized=False,
    )

    return sol.t, sol.y


# -------------------------------------------------------------------
# 2. RMSE entre serie real semanal y simulación
# -------------------------------------------------------------------


def rmse_for_params(beta, c, N0, base_params, real_weeks):
    """
    Calcula el RMSE entre la serie real semanal de casos y la serie simulada
    a partir de parámetros (beta, c, N0) modificando base_params.
    """
    params = base_params.copy()
    params["beta"] = float(beta)
    params["c"] = float(c)
    params["N0"] = float(N0)

    n_weeks = len(real_weeks)
    n_days = int(n_weeks * 7)

    t, y = simulate_seir(params, n_days)
    I = y[2]  # compartimiento infeccioso

    # Pasar de diario a semanal (promedio por semana)
    sim_df = (
        pd.DataFrame({"t": t, "I": I})
        .assign(week=lambda df: (df["t"] // 7).astype(int))
        .groupby("week")["I"]
        .mean()
        .reset_index(drop=True)
    )

    n = min(len(real_weeks), len(sim_df))
    real = real_weeks.iloc[:n].to_numpy(dtype=float)
    sim_I = sim_df.iloc[:n].to_numpy(dtype=float)

    # Escalamiento lineal para comparar picos
    if sim_I.max() > 0:
        k = real.max() / sim_I.max()
    else:
        k = 1.0

    sim_cases = sim_I * k
    rmse = np.sqrt(np.mean((real - sim_cases) ** 2))
    return float(rmse)


# -------------------------------------------------------------------
# 3. Script principal
# -------------------------------------------------------------------


def main() -> None:
    # 0. Configurar experimento en MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("chagas_seir_valledupar")

    # 1. Cargar serie semanal del foco desde chagas_ts_foco.csv
    if not TS_FOCO_PATH.exists():
        raise FileNotFoundError(f"No se encontró {TS_FOCO_PATH}")

    df_ts = pd.read_csv(TS_FOCO_PATH)

    # Columna de casos
    if "casos" not in df_ts.columns:
        raise ValueError(f"Se esperaba columna 'casos' en {TS_FOCO_PATH}")

    # Manejar posibles nombres de fecha: 'Fecha' o 'fecha'
    if "Fecha" in df_ts.columns:
        df_ts["Fecha"] = pd.to_datetime(df_ts["Fecha"])
        df_ts = df_ts.sort_values("Fecha")
        fechas = df_ts["Fecha"].reset_index(drop=True)
    elif "fecha" in df_ts.columns:
        df_ts["Fecha"] = pd.to_datetime(df_ts["fecha"])
        df_ts = df_ts.sort_values("Fecha")
        fechas = df_ts["Fecha"].reset_index(drop=True)
    else:
        # Sin fecha explícita: ordenar por año y semana si existen
        if {"ANO", "SEMANA"}.issubset(df_ts.columns):
            df_ts = df_ts.sort_values(["ANO", "SEMANA"])
        fechas = None

    real_weeks = df_ts["casos"].astype(float).reset_index(drop=True)

    # 2. Cargar parámetros base
    if not PARAMS_BASE_PATH.exists():
        raise FileNotFoundError(f"No se encontró {PARAMS_BASE_PATH}")

    with PARAMS_BASE_PATH.open() as f:
        base_params = json.load(f)

    # Verificar que estén las claves mínimas
    required_keys = ["beta", "alpha", "sigma", "gamma", "mu", "c", "d", "N0"]
    for key in required_keys:
        if key not in base_params:
            raise KeyError(f"Falta el parámetro '{key}' en {PARAMS_BASE_PATH}")

    base_params.setdefault("m_out", 0.0)
    base_params.setdefault("Lambda", base_params["mu"] * base_params["N0"])

    # 3. RMSE con parámetros base (referencia)
    rmse_base = rmse_for_params(
        beta=base_params["beta"],
        c=base_params["c"],
        N0=base_params["N0"],
        base_params=base_params,
        real_weeks=real_weeks,
    )

    # 4. Optimización con L-BFGS-B

    beta0 = base_params["beta"]
    c0 = base_params["c"]
    N0_0 = base_params["N0"]

    bounds = [
        (beta0 * 0.2, beta0 * 5.0),          # beta
        (max(0.001, c0 * 0.2), c0 * 5.0),    # c
        (N0_0 * 0.3, N0_0 * 3.0),            # N0
    ]

    def objective(x):
        beta, c, N0 = x
        return rmse_for_params(beta, c, N0, base_params, real_weeks)

    x0 = np.array([beta0, c0, N0_0], dtype=float)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50},
    )

    if not result.success:
        print("Advertencia: minimize no convergió, se usan parámetros base.")
        beta_opt, c_opt, N0_opt = beta0, c0, N0_0
        rmse_opt = rmse_base
    else:
        beta_opt, c_opt, N0_opt = result.x
        rmse_opt = float(result.fun)

    # 5. Simular con parámetros óptimos para construir la curva de ajuste
    params_opt = base_params.copy()
    params_opt["beta"] = float(beta_opt)
    params_opt["c"] = float(c_opt)
    params_opt["N0"] = float(N0_opt)

    n_weeks = len(real_weeks)
    n_days = int(n_weeks * 7)
    t_opt, y_opt = simulate_seir(params_opt, n_days)
    I_opt = y_opt[2]

    sim_opt = (
        pd.DataFrame({"t": t_opt, "I": I_opt})
        .assign(week=lambda df: (df["t"] // 7).astype(int))
        .groupby("week")["I"]
        .mean()
        .reset_index(drop=True)
    )

    n = min(len(real_weeks), len(sim_opt))
    real_plot = real_weeks.iloc[:n].reset_index(drop=True)
    sim_I = sim_opt.iloc[:n].to_numpy(dtype=float)

    if sim_I.max() > 0:
        k_opt = real_plot.max() / sim_I.max()
    else:
        k_opt = 1.0

    modelo_opt = sim_I * k_opt

    # Fechas para el CSV / figura
    if fechas is not None:
        fechas_plot = fechas.iloc[:n].reset_index(drop=True)
    else:
        fechas_plot = pd.date_range(start="2024-01-01", periods=n, freq="W-MON")

    ajuste_df = pd.DataFrame(
        {
            "Fecha": fechas_plot,
            "casos_reales": real_plot,
            "casos_modelo_opt": modelo_opt,
        }
    )

    # 6. Métricas de regresión
    mse = mean_squared_error(real_plot, modelo_opt)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_plot, modelo_opt)
    r2 = r2_score(real_plot, modelo_opt)

    # 7. Clasificación binaria para métricas tipo ROC / matriz de confusión
    #    Definimos "alta incidencia" como casos >= percentil 75 de la serie real
    threshold = float(np.quantile(real_plot, 0.75))

    y_true = (real_plot >= threshold).astype(int)
    y_score = modelo_opt  # salida continua del modelo
    y_pred = (modelo_opt >= threshold).astype(int)

    # Métricas de clasificación
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_roc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    # 8. Guardar parámetros óptimos y CSV de ajuste
    PARAMS_OPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    params_opt["rmse_base"] = float(rmse_base)
    params_opt["rmse_opt"] = float(rmse_opt)
    params_opt["mse"] = float(mse)
    params_opt["mae"] = float(mae)
    params_opt["r2"] = float(r2)
    params_opt["threshold_alta_incidencia"] = threshold

    with PARAMS_OPT_PATH.open("w", encoding="utf-8") as f:
        json.dump(params_opt, f, indent=4, ensure_ascii=False)

    AJUSTE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ajuste_df.to_csv(AJUSTE_CSV_PATH, index=False)

    # 9. Figura comparativa
    plt.figure(figsize=(8, 4))
    plt.plot(
        ajuste_df["Fecha"],
        ajuste_df["casos_reales"],
        marker="o",
        label="Casos reales",
    )
    plt.plot(
        ajuste_df["Fecha"],
        ajuste_df["casos_modelo_opt"],
        marker="x",
        label="Modelo SEIR optimizado",
    )
    plt.xlabel("Fecha")
    plt.ylabel("Casos")
    plt.title(
        f"Valledupar – Ajuste SEIR (RMSE base={rmse_base:.2f}, RMSE opt={rmse_opt:.2f})"
    )
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    AJUSTE_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(AJUSTE_FIG_PATH, dpi=300)
    plt.close()

    # 10. Gráfico de residuos
    residuos = real_plot - modelo_opt
    plt.figure(figsize=(8, 4))
    plt.scatter(ajuste_df["casos_modelo_opt"], residuos)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Casos modelo SEIR")
    plt.ylabel("Residuo (real - modelo)")
    plt.title("Residuos del ajuste SEIR – Valledupar")
    plt.tight_layout()
    plt.savefig(RESID_FIG_PATH, dpi=300)
    plt.close()

    # 11. Curva ROC
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC – Semana de alta incidencia vs normal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROC_FIG_PATH, dpi=300)
    plt.close()

    # 12. Matriz de confusión
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Normal (0)", "Alta (1)"])
    plt.yticks([0, 1], ["Normal (0)", "Alta (1)"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión – Alta incidencia")
    # Etiquetas numéricas
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(CM_FIG_PATH, dpi=300)
    plt.close()

    print(f"Parámetros óptimos guardados en: {PARAMS_OPT_PATH}")
    print(f"Ajuste semanal guardado en: {AJUSTE_CSV_PATH}")
    print(f"Figura de ajuste guardada en: {AJUSTE_FIG_PATH}")
    print(f"Figura de residuos guardada en: {RESID_FIG_PATH}")
    print(f"Curva ROC guardada en: {ROC_FIG_PATH}")
    print(f"Matriz de confusión guardada en: {CM_FIG_PATH}")
    print(f"RMSE base: {rmse_base:.4f} | RMSE óptimo: {rmse_opt:.4f}")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    print(
        f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {f1:.4f} | AUC ROC: {auc_roc:.4f}"
    )

    # 13. Registro en MLflow
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="calibracion_seir_valledupar"):

            # Hiperparámetros / parámetros del modelo
            mlflow.log_param("municipio", "Valledupar")
            mlflow.log_param("beta_base", float(beta0))
            mlflow.log_param("c_base", float(c0))
            mlflow.log_param("N0_base", float(N0_0))
            mlflow.log_param("beta_opt", float(beta_opt))
            mlflow.log_param("c_opt", float(c_opt))
            mlflow.log_param("N0_opt", float(N0_opt))
            mlflow.log_param("sigma", float(base_params["sigma"]))
            mlflow.log_param("gamma", float(base_params["gamma"]))
            mlflow.log_param("mu", float(base_params["mu"]))
            mlflow.log_param("alpha", float(base_params["alpha"]))
            mlflow.log_param("d", float(base_params["d"]))
            mlflow.log_param("m_out", float(base_params.get("m_out", 0.0)))
            mlflow.log_param("Lambda", float(base_params["Lambda"]))
            mlflow.log_param("n_weeks", int(n_weeks))
            mlflow.log_param("threshold_alta_incidencia", threshold)

            # Métricas de rendimiento
            mlflow.log_metric("rmse_base", float(rmse_base))
            mlflow.log_metric("rmse_opt", float(rmse_opt))
            mlflow.log_metric("mse", float(mse))
            mlflow.log_metric("mae", float(mae))
            mlflow.log_metric("r2", float(r2))
            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("precision", float(precision))
            mlflow.log_metric("recall", float(recall))
            mlflow.log_metric("f1", float(f1))
            mlflow.log_metric("auc_roc", float(auc_roc))

            # Artefactos
            mlflow.log_artifact(str(PARAMS_OPT_PATH), artifact_path="models")
            mlflow.log_artifact(str(AJUSTE_CSV_PATH), artifact_path="reports")
            mlflow.log_artifact(str(AJUSTE_FIG_PATH), artifact_path="figures")
            mlflow.log_artifact(str(RESID_FIG_PATH), artifact_path="figures")
            mlflow.log_artifact(str(ROC_FIG_PATH), artifact_path="figures")
            mlflow.log_artifact(str(CM_FIG_PATH), artifact_path="figures")

    else:
        print("MLflow no está instalado; se omite el registro de experimentos.")


if __name__ == "__main__":
    main()