"""
prepare_data.py

Construye el dataset agregado a nivel semanal y la serie del foco epidemiológico.
Produce:
- data/clean/chagas_prepared.csv
- data/clean/chagas_ts_foco.csv

Este script está alineado con los Notebooks 02 y 03 del proyecto.
"""

import pandas as pd
from pathlib import Path
from datetime import date


# ------------------------- RUTAS -------------------------

INPUT_PATH = Path("data/clean/chagas_clean.csv")
OUT_PREPARED = Path("data/clean/chagas_prepared.csv")
OUT_TS_FOCO = Path("data/clean/chagas_ts_foco.csv")

# Foco por defecto (Valledupar, Cesar)
DEP_FOCO = "CESAR"
MPIO_FOCO = "VALLEDUPAR"


# ------------------------- FUNCIONES -------------------------

def build_date_from_year_week(year: int, week: int) -> date:
    """Construye el lunes ISO de la semana epidemiológica año-semana."""
    try:
        return date.fromisocalendar(int(year), int(week), 1)
    except Exception:
        # Si hay semanas fuera de rango, las forzamos a semana 1
        return date.fromisocalendar(int(year), 1, 1)


def ensure_cases_column(df: pd.DataFrame) -> str:
    """Garantiza que exista una columna 'casos'."""
    if "casos" in df.columns:
        return "casos"

    candidates = [c for c in df.columns if "caso" in c.lower()]
    if candidates:
        col = candidates[0]
        df.rename(columns={col: "casos"}, inplace=True)
        print(f"Usando columna '{col}' como 'casos'.")
        return "casos"

    df["casos"] = 1
    print("No se encontró columna de casos; se asume 1 caso por fila.")
    return "casos"


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        "ANO",
        "SEMANA",
        "Departamento_residencia",
        "Municipio_residencia",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


# ------------------------- MAIN -------------------------

def main() -> None:
    print("Leyendo archivo limpio:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    validate_columns(df)

    # Asegurar columna casos
    cases_col = ensure_cases_column(df)
    df["casos"] = pd.to_numeric(df[cases_col], errors="coerce").fillna(0).astype(int)

    # Normalizar año y semana
    df["ANO"] = df["ANO"].astype(int)
    df["SEMANA"] = df["SEMANA"].astype(int)

    # Crear columna fecha
    df["Fecha"] = df.apply(
        lambda r: build_date_from_year_week(r["ANO"], r["SEMANA"]),
        axis=1,
    )

    # Normalizar texto
    df["Departamento_residencia"] = df["Departamento_residencia"].astype(str).str.upper().str.strip()
    df["Municipio_residencia"] = df["Municipio_residencia"].astype(str).str.upper().str.strip()

    # -------------------------
    # 1) Dataset agregado semanal
    # -------------------------
    group_cols = [
        "ANO",
        "SEMANA",
        "Fecha",
        "Departamento_residencia",
        "Municipio_residencia",
    ]

    prepared = (
        df.groupby(group_cols, as_index=False)["casos"]
        .sum()
        .sort_values(group_cols)
    )

    OUT_PREPARED.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUT_PREPARED, index=False)
    print(f"Dataset semanal agregado guardado en: {OUT_PREPARED}")


    # -------------------------
    # 2) Serie del foco (Valledupar)
    # -------------------------

    foco = (
        prepared[
            (prepared["Departamento_residencia"] == DEP_FOCO)
            & (prepared["Municipio_residencia"] == MPIO_FOCO)
        ]
        .copy()
        .sort_values(["ANO", "SEMANA"])
    )

    foco_ts = foco[["Fecha", "ANO", "SEMANA", "casos"]]
    foco_ts.to_csv(OUT_TS_FOCO, index=False)

    print(f"Serie temporal del foco guardada en: {OUT_TS_FOCO}")


if __name__ == "__main__":
    main()