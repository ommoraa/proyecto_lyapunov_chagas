import pandas as pd
from pathlib import Path
from datetime import date


def build_date_from_year_week(year: int, week: int) -> date:
    """
    Construye una fecha (lunes de cada semana ISO) a partir de año y semana.
    """
    return date.fromisocalendar(int(year), int(week), 1)


def ensure_cases_column(df: pd.DataFrame) -> str:
    """
    Garantiza que exista una columna 'casos' en el DataFrame.

    - Si ya hay una columna llamada 'casos', se usa directamente.
    - Si hay otra columna cuyo nombre contenga 'caso', se renombra a 'casos'.
    - Si no hay ninguna columna relacionada, se crea 'casos' = 1 (cada fila es un caso).
    """
    # 1) Columna exactamente 'casos'
    if "casos" in df.columns:
        return "casos"

    # 2) Otras variantes con la palabra 'caso'
    candidates = [c for c in df.columns if "caso" in c.lower()]
    if candidates:
        col = candidates[0]
        df.rename(columns={col: "casos"}, inplace=True)
        print(f"Usando columna '{col}' como 'casos'.")
        return "casos"

    # 3) No hay ninguna columna: cada fila es un caso
    df["casos"] = 1
    print("No se encontró columna de casos; se asume 1 caso por fila.")
    return "casos"


def main() -> None:
    input_path = Path("data/clean/chagas_clean.csv")
    output_path = Path("data/clean/chagas_prepared.csv")

    df = pd.read_csv(input_path)

    # Asegurar columna de casos
    cases_col = ensure_cases_column(df)
    df["casos"] = pd.to_numeric(df[cases_col], errors="coerce").fillna(0).astype(int)

    # Normalizar año y semana
    df["ANO"] = df["ANO"].astype(int)
    df["SEMANA"] = df["SEMANA"].astype(int)

    # Crear columna Fecha si no existe
    if "Fecha" not in df.columns:
        df["Fecha"] = df.apply(
            lambda r: build_date_from_year_week(r["ANO"], r["SEMANA"]),
            axis=1,
        )

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
        .sort_values(
            ["ANO", "SEMANA", "Departamento_residencia", "Municipio_residencia"]
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(f"Archivo preparado guardado en: {output_path}")


if __name__ == "__main__":
    main()