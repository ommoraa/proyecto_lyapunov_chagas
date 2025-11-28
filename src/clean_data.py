from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/Datos_2024_205.xlsx")
OUT_PATH = Path("data/clean/chagas_clean.csv")


def main() -> None:
    df = pd.read_excel(RAW_PATH)

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

    df = df[cols].copy()

    df["FEC_NOT"] = pd.to_datetime(df["FEC_NOT"], errors="coerce")

    df = df.dropna(subset=["FEC_NOT"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()