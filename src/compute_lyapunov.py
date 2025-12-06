"""
compute_lyapunov.py

Cálculo del exponente de Lyapunov máximo para el modelo SEIR continuo
con incidencia no lineal y migración, usando los parámetros calibrados
para Valledupar (Cesar).

Este script:
- Lee parámetros óptimos desde models/seir_params_opt.json
- Simula dos trayectorias cercanas del sistema SEIR (S, E, I, R, V)
- Aplica un esquema tipo Benettin para estimar el exponente de Lyapunov máximo
- Guarda:
    - models/lyapunov_valledupar.json
    - reports/lyapunov_distances_valledupar.csv
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# --------------------- RUTAS ---------------------

PARAMS_OPT_PATH = Path("models/seir_params_opt.json")
OUT_JSON = Path("models/lyapunov_valledupar.json")
OUT_DIST_CSV = Path("reports/lyapunov_distances_valledupar.csv")


# --------------------- PREPARACIÓN DE PARÁMETROS ---------------------

def prepare_params(params: dict) -> dict:
    """
    Asegura que todos los parámetros necesarios estén presentes,
    asignando valores por defecto en caso necesario.
    """
    p = params.copy()

    p_clean = {
        "N0": float(p.get("N0", 3500.0)),
        "beta": float(p.get("beta", 0.2)),
        "alpha": float(p.get("alpha", 0.01)),
        "sigma": float(p.get("sigma", 1 / 60)),
        "gamma": float(p.get("gamma", 1 / 180)),
        "mu": float(p.get("mu", 0.000039)),
        "c": float(p.get("c", 0.05)),
        "d": float(p.get("d", 1 / 365)),
        "m_out": float(p.get("m_out", 0.017)),
    }

    # Tasa de entrada / nacimiento
    p_clean["Lambda"] = float(p.get("Lambda", p_clean["mu"] * p_clean["N0"]))

    return p_clean


# --------------------- SISTEMA SEIR ---------------------

def seir_system(t, y, p):
    """
    Sistema SEIR con incidencia no lineal y migración.
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
    m_out = p["m_out"]
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


# --------------------- INTEGRACIÓN DE UNA TRAYECTORIA ---------------------

def integrate_trajectory(y0, p, t0, t1, dt=1.0):
    """
    Integra el sistema SEIR desde t0 hasta t1 con paso aproximado dt.
    Devuelve (t, Y) donde Y es un arreglo 2D con columnas = estados en el tiempo.
    """
    t_eval = np.arange(t0, t1 + dt, dt)

    sol = solve_ivp(
        lambda t, y: seir_system(t, y, p),
        (t0, t1),
        y0,
        t_eval=t_eval,
        vectorized=False,
    )
    return sol.t, sol.y


# --------------------- CÁLCULO DEL EXPONENTE DE LYAPUNOV ---------------------

def lyapunov_maximum(params: dict, T_total: float = 365.0,
                     dt_block: float = 7.0, delta0: float = 1e-5):
    """
    Calcula el exponente de Lyapunov máximo usando un esquema tipo Benettin.

    params   : diccionario de parámetros (ya calibrados)
    T_total  : tiempo total de integración (días)
    dt_block : tamaño de cada bloque de integración (días)
    delta0   : norma inicial de la perturbación
    """
    p = prepare_params(params)

    N0 = p["N0"]
    # Condiciones iniciales base (coherentes con simulate_seir)
    I0 = 1.0
    E0 = 0.0
    R0 = 0.0
    V0 = 1.0
    S0 = N0 - I0 - E0 - R0

    y1 = np.array([S0, E0, I0, R0, V0], dtype=float)

    # Perturbación inicial en la componente infecciosa, por ejemplo
    v0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
    v0 = v0 / np.linalg.norm(v0) * delta0
    y2 = y1 + v0

    n_blocks = int(T_total / dt_block)
    sum_logs = 0.0

    distances = []
    times = []

    t_current = 0.0

    for k in range(n_blocks):
        t_next = t_current + dt_block

        # Integrar trayectoria de referencia
        _, Y1 = integrate_trajectory(y1, p, t_current, t_next, dt=1.0)
        # Integrar trayectoria perturbada
        _, Y2 = integrate_trajectory(y2, p, t_current, t_next, dt=1.0)

        # Estado final de cada trayectoria
        y1_final = Y1[:, -1]
        y2_final = Y2[:, -1]

        # Distancia entre trayectorias
        diff = y2_final - y1_final
        d = np.linalg.norm(diff)
        if d == 0.0:
            d = 1e-16  # evitar problemas numéricos

        sum_logs += np.log(d / delta0)

        distances.append(d)
        times.append(t_next)

        # Renormalizar perturbación alrededor de la trayectoria de referencia
        v = diff / d * delta0
        y1 = y1_final
        y2 = y1_final + v

        t_current = t_next

    T = n_blocks * dt_block
    lambda_max = sum_logs / T

    results = pd.DataFrame({"t": times, "distancia": distances})

    return float(lambda_max), results


# --------------------- SCRIPT PRINCIPAL ---------------------

def main():
    print("Leyendo parámetros calibrados desde:", PARAMS_OPT_PATH)
    with PARAMS_OPT_PATH.open() as f:
        params_opt = json.load(f)

    # Horizonte de integración (365 días por coherencia con la simulación)
    T_total = 365.0
    dt_block = 7.0
    delta0 = 1e-5

    lambda_max, dist_df = lyapunov_maximum(
        params_opt,
        T_total=T_total,
        dt_block=dt_block,
        delta0=delta0,
    )

    print(f"Exponente de Lyapunov máximo estimado: {lambda_max:.6f}")

    # Guardar distancias en CSV
    OUT_DIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(OUT_DIST_CSV, index=False)
    print("Evolución de distancias guardada en:", OUT_DIST_CSV)

    # Guardar resultado en JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "municipio": "Valledupar",
        "lambda_max": float(lambda_max),
        "T_total_days": float(T_total),
        "dt_block_days": float(dt_block),
        "delta0": float(delta0),
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("Resultado de Lyapunov guardado en:", OUT_JSON)


if __name__ == "__main__":
    main()