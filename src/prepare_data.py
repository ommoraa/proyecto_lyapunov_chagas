from pathlib import Path

import pandas as pd


IN_PATH = Path("data/clean/chagas_clean.csv")
OUT_PATH = Path("data/clean/chagas_prepared.csv")


def main() -> None:
    df = pd.read_csv(IN_PATH)

    group_cols = [
        "ANO",
        "SEMANA",
        "Departamento_residencia",
        "Municipio_residencia",
    ]

    agg = (
        df.groupby(group_cols)
        .size()
        .reset_index(name="casos")
        .sort_values(group_cols)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()