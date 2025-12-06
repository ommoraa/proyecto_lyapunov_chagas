"""
simulate_seir.py

Simula el modelo SEIR con incidencia no lineal y migración durante 365 días.
Usa los parámetros base definidos en models/seir_params.json.

Produce:
- reports/seir_simulation.csv

Este script está alineado con los Notebooks 03 y 04.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


# --------------------- RUTAS ---------------------

PARAMS_PATH = Path("models/seir_params.json")
OUT_CSV = Path("reports/seir_simulation.csv")


# --------------------- SISTEMA SEIR ---------------------

def prepare_params(params):
    """
    Asegura que todos los parámetros necesarios estén presentes,
    asignando valores por defecto en caso necesario.
    """
    p = params.copy()

    p_clean = {
        "N0": float(p.get("N0", 3500.0)),
        "beta": float(p.get("beta", 0.2)),
        "alpha": float(p.get("alpha", 0.01)),
        "sigma": float(p.get("sigma", 1/60)),
        "gamma": float(p.get("gamma", 1/180)),
        "mu": float(p.get("mu", 0.000039)),
        "c": float(p.get("c", 0.05)),
        "d": float(p.get("d", 1/365)),
        "m_out": float(p.get("m_out", 0.017)),
    }

    # Tasa de entrada / nacimiento
    p_clean["Lambda"] = p.get("Lambda", p_clean["mu"] * p_clean["N0"])

    return p_clean


def seir_system(t, y, p):
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
        return [0, 0, 0, 0, 0]

    infection = beta * S * V / (1 + alpha * I)

    dSdt = Lambda - infection - (mu + m_out) * S
    dEdt = infection - (sigma + mu + m_out) * E
    dIdt = sigma * E - (gamma + mu + m_out) * I
    dRdt = gamma * I - (mu + m_out) * R
    dVdt = c * I - d * V

    return [dSdt, dEdt, dIdt, dRdt, dVdt]


# --------------------- SIMULACIÓN ---------------------

def main():
    print("Leyendo parámetros desde:", PARAMS_PATH)

    with PARAMS_PATH.open() as f:
        params_raw = json.load(f)

    p = prepare_params(params_raw)

    print("Parámetros usados para la simulación:")
    print(json.dumps(p, indent=4))

    N0 = p["N0"]
    I0 = 1.0
    E0 = 0.0
    R0 = 0.0
    V0 = 1.0
    S0 = N0 - I0 - E0 - R0

    y0 = [S0, E0, I0, R0, V0]

    # Simulación diaria
    t_span = (0.0, 365.0)
    t_eval = np.arange(0, 366, 1)

    sol = solve_ivp(
        lambda t, y: seir_system(t, y, p),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
    )

    df = pd.DataFrame({
        "t": sol.t,
        "S": sol.y[0],
        "E": sol.y[1],
        "I": sol.y[2],
        "R": sol.y[3],
        "V": sol.y[4],
    })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print("Simulación guardada en:", OUT_CSV)
    print("Filas generadas:", len(df))


if __name__ == "__main__":
    main()