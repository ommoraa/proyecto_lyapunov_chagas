from pathlib import Path
import json

import pandas as pd


# Municipio foco
DEP = "CESAR"
MPIO = "VALLEDUPAR"

CLEAN_PATH = Path("data/clean/chagas_clean.csv")
AGG_PATH = Path("data/clean/chagas_prepared.csv")
OUT_PATH = Path("models/seir_params.json")


def estimate_migration_rate(df_clean: pd.DataFrame) -> float:
    """Tasa simple de migración de salida para el municipio foco."""
    df_res = df_clean[
        (df_clean["Departamento_residencia"] == DEP)
        & (df_clean["Municipio_residencia"] == MPIO)
    ].copy()

    if df_res.empty:
        return 0.0

    migrados = df_res[
        df_res["Municipio_ocurrencia"] != df_res["Municipio_residencia"]
    ]

    tasa = len(migrados) / len(df_res)

    return float(tasa)


def main() -> None:
    df_clean = pd.read_csv(CLEAN_PATH)
    df_agg = pd.read_csv(AGG_PATH)

    foco = df_agg[
        (df_agg["Departamento_residencia"] == DEP)
        & (df_agg["Municipio_residencia"] == MPIO)
    ].copy()

    foco = foco.sort_values(["ANO", "SEMANA"])

    if foco.empty:
        raise ValueError("No hay datos agregados para el municipio foco.")

    max_casos = foco["casos"].max()
    total_casos = foco["casos"].sum()

    N0 = float(max(max_casos * 100.0, total_casos * 10.0))

    m_out = estimate_migration_rate(df_clean)

    params = {
        "beta": 0.2,
        "alpha": 0.01,
        "sigma": 1.0 / 60.0,
        "gamma": 1.0 / 180.0,
        "mu": 1.0 / (70.0 * 365.0),
        "c": 0.05,
        "d": 1.0 / 365.0,
        "N0": N0,
        "m_out": m_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()