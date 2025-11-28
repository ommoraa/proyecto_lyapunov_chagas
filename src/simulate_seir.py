from pathlib import Path

import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


PARAMS_PATH = Path("models/seir_params.json")
OUT_CSV = Path("reports/seir_simulation.csv")


def seir_system(t, y, p):
    S, E, I, R, V = y
    beta = p["beta"]
    alpha = p["alpha"]
    sigma = p["sigma"]
    gamma = p["gamma"]
    mu = p["mu"]
    c = p["c"]
    d = p["d"]

    N = S + E + I + R
    infection = beta * S * V / (1.0 + alpha * I)

    dSdt = p["Lambda"] - infection - mu * S
    dEdt = infection - (sigma + mu) * E
    dIdt = sigma * E - (gamma + mu) * I
    dRdt = gamma * I - mu * R
    dVdt = c * I - d * V

    return [dSdt, dEdt, dIdt, dRdt, dVdt]


def main() -> None:
    with PARAMS_PATH.open() as f:
        params = json.load(f)

    N0 = params.get("N0", 1_000.0)
    I0 = 1.0
    E0 = 0.0
    R0 = 0.0
    V0 = 1.0
    S0 = N0 - I0 - E0 - R0

    params.setdefault("Lambda", params["mu"] * N0)

    t_span = (0.0, 365.0)
    t_eval = np.linspace(t_span[0], t_span[1], 366)

    sol = solve_ivp(
        lambda t, y: seir_system(t, y, params),
        t_span=t_span,
        y0=[S0, E0, I0, R0, V0],
        t_eval=t_eval,
        vectorized=False,
    )

    df = pd.DataFrame(
        {
            "t": sol.t,
            "S": sol.y[0],
            "E": sol.y[1],
            "I": sol.y[2],
            "R": sol.y[3],
            "V": sol.y[4],
        }
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()