import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# MLflow opcional: si no está instalado, el script sigue funcionando
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


# -------------------------
# Modelo SEIR
# -------------------------


def simulate_seir(beta, sigma, gamma, N, I0, E0, R0, n_steps):
    """
    Simula un modelo SEIR sencillo usando un esquema de Euler hacia adelante
    con paso de tiempo dt = 1 (por ejemplo, 1 semana).
    """
    S0 = N - I0 - E0 - R0

    S = np.zeros(n_steps)
    E = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)

    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    dt = 1.0

    for t in range(1, n_steps):
        dS = -beta * S[t - 1] * I[t - 1] / N
        dE = beta * S[t - 1] * I[t - 1] / N - sigma * E[t - 1]
        dI = sigma * E[t - 1] - gamma * I[t - 1]
        dR = gamma * I[t - 1]

        S[t] = S[t - 1] + dt * dS
        E[t] = E[t - 1] + dt * dE
        I[t] = I[t - 1] + dt * dI
        R[t] = R[t - 1] + dt * dR

    return S, E, I, R


def rmse(params, casos_reales, N, I0, E0, R0):
    """
    Función objetivo: RMSE entre la serie simulada de infecciosos (I)
    y la serie de casos observados.
    """
    beta, sigma, gamma = params

    # Evitar parámetros no físicos
    if beta <= 0 or sigma <= 0 or gamma <= 0:
        return 1e9

    n_steps = len(casos_reales)
    _, _, I_sim, _ = simulate_seir(beta, sigma, gamma, N, I0, E0, R0, n_steps)

    # Para evitar valores negativos por discretización
    I_sim = np.clip(I_sim, a_min=0, a_max=None)

    return np.sqrt(np.mean((I_sim - casos_reales) ** 2))


# -------------------------
# Script principal
# -------------------------


def main():
    base_dir = Path(__file__).resolve().parents[1]

    data_path = base_dir / "data" / "clean" / "chagas_prepared.csv"
    params_path = base_dir / "models" / "seir_params.json"
    sim_path = base_dir / "reports" / "seir_simulation.csv"

    # 1. Cargar datos preparados
    df = pd.read_csv(data_path)

    # Filtrar municipio de interés
    municipio_objetivo = "VALLEDUPAR"
    df_mun = df[
        df["Municipio_residencia"].astype(str).str.upper() == municipio_objetivo
    ].copy()

    if df_mun.empty:
        raise ValueError(f"No se encontraron registros para {municipio_objetivo} en {data_path}")

    # Asegurar que haya columna Fecha; si no, usamos un índice simple
    if "Fecha" in df_mun.columns:
        df_mun = df_mun.sort_values("Fecha")
        fechas = pd.to_datetime(df_mun["Fecha"])
    else:
        df_mun = df_mun.sort_values(["ANO", "SEMANA"])
        fechas = pd.to_datetime(df_mun["ANO"].astype(str) + "-01-01") + pd.to_timedelta(
            (df_mun["SEMANA"] - 1) * 7, unit="D"
        )

    casos = df_mun["casos"].astype(float).values
    n_steps = len(casos)

    # 2. Definir condiciones iniciales y parámetros de población
    N = 100_000  # población efectiva aproximada
    I0 = max(1.0, casos[0])
    E0 = I0
    R0 = 0.0

    # 3. Optimización de parámetros (beta, sigma, gamma)
    x0 = np.array([0.5, 0.2, 0.1])  # valores iniciales razonables
    bounds = [(1e-6, 2.0), (1e-6, 2.0), (1e-6, 2.0)]

    result = minimize(
        rmse,
        x0,
        args=(casos, N, I0, E0, R0),
        method="L-BFGS-B",
        bounds=bounds,
    )

    beta_opt, sigma_opt, gamma_opt = result.x
    rmse_final = rmse(result.x, casos, N, I0, E0, R0)

    # 4. Simulación final con parámetros óptimos
    S_sim, E_sim, I_sim, R_sim = simulate_seir(
        beta_opt, sigma_opt, gamma_opt, N, I0, E0, R0, n_steps
    )
    I_sim = np.clip(I_sim, a_min=0, a_max=None)

    # 5. Guardar parámetros en JSON
    params = {
        "municipio": municipio_objetivo.title(),
        "beta": float(beta_opt),
        "sigma": float(sigma_opt),
        "gamma": float(gamma_opt),
        "N": int(N),
        "rmse": float(rmse_final),
    }

    params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)

    # 6. Guardar simulación en CSV
    sim_path.parent.mkdir(parents=True, exist_ok=True)
    df_sim = pd.DataFrame(
        {
            "Fecha": fechas,
            "casos_reales": casos,
            "S": S_sim,
            "E": E_sim,
            "I_sim": I_sim,
            "R": R_sim,
        }
    )
    df_sim.to_csv(sim_path, index=False)

    print(f"Parámetros óptimos guardados en: {params_path}")
    print(f"Simulación SEIR guardada en: {sim_path}")
    print(f"RMSE final: {rmse_final:.4f}")

    # 7. Registro en MLflow (si está disponible)
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("chagas_seir_valledupar")
        with mlflow.start_run(run_name="calibracion_seir_valledupar"):
            # Parámetros
            mlflow.log_param("municipio", municipio_objetivo.title())
            mlflow.log_param("N", int(N))
            mlflow.log_param("beta", float(beta_opt))
            mlflow.log_param("sigma", float(sigma_opt))
            mlflow.log_param("gamma", float(gamma_opt))

            # Métricas
            mlflow.log_metric("rmse", float(rmse_final))

            # Artefactos principales
            mlflow.log_artifact(str(params_path), artifact_path="models")
            mlflow.log_artifact(str(sim_path), artifact_path="reports")

            # Figura simple de ajuste observados vs simulados
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df_sim["Fecha"], df_sim["casos_reales"], label="Casos reales")
                ax.plot(df_sim["Fecha"], df_sim["I_sim"], label="Infecciosos simulados")
                ax.set_xlabel("Fecha")
                ax.set_ylabel("Casos")
                ax.legend()
                ax.grid(True, alpha=0.3)

                mlflow.log_figure(fig, "figures/ajuste_seir_valledupar.png")
                plt.close(fig)
            except Exception as e:  # pragma: no cover
                print(f"No fue posible registrar la figura en MLflow: {e}")
    else:
        print("MLflow no está instalado; se omite el registro de experimentos.")


if __name__ == "__main__":
    main()