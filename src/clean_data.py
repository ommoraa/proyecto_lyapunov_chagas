"""
clean_data.py

Limpieza inicial del dataset de Chagas 2024.
Transforma datos brutos desde Excel y genera chagas_clean.csv.

Este script está alineado con el Notebook 01 (EDA) y es reproducible vía DVC.
"""

from pathlib import Path
import pandas as pd


# ----------------- Rutas -----------------
RAW_PATH = Path("data/raw/Datos_2024_205.xlsx")
OUT_PATH = Path("data/clean/chagas_clean.csv")


# ----------------- Funciones -----------------
def validate_columns(df: pd.DataFrame, required_cols: list) -> None:
    """Verifica que las columnas requeridas existan."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el archivo RAW: {missing}")


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza estructural básica al archivo de entrada."""

    # Columnas a mantener (definidas por el Notebook 1)
    cols = [
        "CONSECUTIVE",
        "COD_EVE",
        "FEC_NOT",
        "SEMANA",
        "ANO",
        "EDAD",
        "UNI_MED",
        "Departamento_residencia",
        "Municipio_residencia",
        "Departamento_ocurrencia",
        "Municipio_ocurrencia",
        "Departamento_Notificacion",
        "Municipio_notificacion",
    ]

    validate_columns(df, cols)

    df = df[cols].copy()

    # Normalizar mayúsculas/minúsculas
    str_cols = [
        "Departamento_residencia",
        "Municipio_residencia",
        "Departamento_ocurrencia",
        "Municipio_ocurrencia",
        "Departamento_Notificacion",
        "Municipio_notificacion",
    ]
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Convertir fechas
    df["FEC_NOT"] = pd.to_datetime(df["FEC_NOT"], errors="coerce")

    # Quitar registros sin fecha
    df = df.dropna(subset=["FEC_NOT"])

    # Asegurar tipos numéricos
    numeric_cols = ["SEMANA", "ANO", "EDAD"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["SEMANA", "ANO"])

    return df


# ----------------- Ejecución principal -----------------
def main() -> None:
    print("Leyendo archivo RAW:", RAW_PATH)

    df_raw = pd.read_excel(RAW_PATH)
    df_clean = clean_raw_data(df_raw)

    print("Filas después de limpieza:", len(df_clean))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    print("Archivo generado:", OUT_PATH)


if __name__ == "__main__":
    main()