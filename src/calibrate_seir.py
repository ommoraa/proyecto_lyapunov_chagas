from pathlib import Path

import json
import pandas as pd


IN_PATH = Path("data/clean/chagas_prepared.csv")
OUT_PATH = Path("models/seir_params.json")


def main() -> None:
    df = pd.read_csv(IN_PATH)

    params = {
        "beta": 0.2,
        "alpha": 0.01,
        "sigma": 1 / 60,
        "gamma": 1 / 180,
        "mu": 1 / (70 * 365),
        "c": 0.05,
        "d": 1 / 365,
        "N0": float(df["casos"].max() * 100) if not df.empty else 1000.0,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()