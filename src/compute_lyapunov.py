import json
import numpy as np
import pandas as pd
from pathlib import Path


# -------------------------
# Dinámica SEIR
# -------------------------

def seir_rhs(state, beta, sigma, gamma, N):
    """
    Ecuaciones diferenciales del modelo SEIR en forma discreta
    (un paso de integración con tamaño dt).
    """
    S, E, I, R = state

    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return np.array([dS, dE, dI, dR], dtype=float)


def jacobian_seir(state, beta, sigma, gamma, N):
    """
    Matriz jacobiana del modelo SEIR evaluada en el estado actual.
    """
    S, E, I, R = state

    J = np.zeros((4, 4), dtype=float)

    # d(dS)/dS, d(dS)/dI
    J[0, 0] = -beta * I / N
    J[0, 2] = -beta * S / N

    # d(dE)/dS, d(dE)/dE, d(dE)/dI
    J[1, 0] = beta * I / N
    J[1, 1] = -sigma
    J[1, 2] = beta * S / N

    # d(dI)/dE, d(dI)/dI
    J[2, 1] = sigma
    J[2, 2] = -gamma

    # d(dR)/dI
    J[3, 2] = gamma

    return J


# -------------------------
# Cálculo del exponente de Lyapunov máximo
# -------------------------

def largest_lyapunov(beta, sigma, gamma, N, x0,
                     n_steps=200, dt=1.0, renorm_every=5):
    """
    Cálculo aproximado del exponente de Lyapunov máximo usando un
    esquema tipo Benettin para sistemas continuos.

    Parámetros
    ----------
    beta, sigma, gamma : float
        Parámetros del modelo SEIR.
    N : float
        Tamaño poblacional efectivo.
    x0 : iterable de longitud 4
        Estado inicial [S0, E0, I0, R0].
    n_steps : int
        Número de pasos de integración.
    dt : float
        Tamaño de paso.
    renorm_every : int
        Cada cuántos pasos se renormaliza el vector de perturbación.
    """
    x = np.array(x0, dtype=float)

    # Perturbación inicial aleatoria normalizada
    v = np.random.randn(4)
    v /= np.linalg.norm(v)

    sum_log = 0.0
    t = 0.0

    for k in range(1, n_steps + 1):
        # Paso del sistema principal
        dx = seir_rhs(x, beta, sigma, gamma, N)
        x = x + dt * dx

        # Paso del sistema variacional usando el jacobiano
        J = jacobian_seir(x, beta, sigma, gamma, N)
        v = v + dt * (J @ v)

        # Renormalización periódica
        if k % renorm_every == 0:
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                # Evitar vectores nulos por redondeo numérico
                v = np.random.randn(4)
                v /= np.linalg.norm(v)
            else:
                sum_log += np.log(norm_v)
                v /= norm_v

        t += dt

    # Exponente de Lyapunov máximo aproximado
    lyap_max = sum_log / (t if t > 0 else 1.0)
    return lyap_max


# -------------------------
# Utilidades para cargar datos y parámetros
# -------------------------

def load_series_for_municipio(path_csv, municipio):
    """
    Carga la serie de casos para un municipio específico desde un CSV limpio.
    No exige columna Fecha. Si existe, la usa para ordenar; si no, sigue tal cual.
    """
    df = pd.read_csv(path_csv)

    # --- Detectar columna municipio ---
    posibles_municipio = ["Municipio_residencia", "municipio", "Municipio"]
    col_mpio = next((c for c in posibles_municipio if c in df.columns), None)
    if col_mpio is None:
        raise ValueError(f"No se encontró columna de municipio. Columnas: {df.columns.tolist()}")

    df_mun = df[df[col_mpio].astype(str).str.upper() == municipio.upper()].copy()

    # --- Ordenar por fecha SOLO SI existe ---
    if "Fecha" in df_mun.columns:
        df_mun = df_mun.sort_values("Fecha")

    # --- Detectar columna casos ---
    posibles_casos = ["casos", "Casos", "n_casos"]
    col_casos = next((c for c in posibles_casos if c in df_mun.columns), None)
    if col_casos is None:
        raise ValueError(f"No se encontró columna de casos. Columnas filtradas: {df_mun.columns.tolist()}")

    return df_mun[col_casos].values.astype(float)


def load_seir_params(path_json):
    """
    Carga parámetros optimizados del modelo SEIR desde un archivo JSON.

    Se espera que el JSON tenga las claves:
    - 'beta'
    - 'sigma'
    - 'gamma'
    """
    with open(path_json, "r", encoding="utf-8") as f:
        params = json.load(f)

    beta = params["beta"]
    sigma = params["sigma"]
    gamma = params["gamma"]
    return beta, sigma, gamma


# -------------------------
# Script principal
# -------------------------

def main():
    # Directorio raíz del proyecto
    base_dir = Path(__file__).resolve().parents[1]

    data_path = base_dir / "data" / "clean" / "chagas_prepared.csv"
    params_path = base_dir / "models" / "seir_params.json"
    output_path = base_dir / "models" / "lyapunov_valledupar.json"

    # 1. Cargar serie para definir el horizonte de integración
    serie_valledupar = load_series_for_municipio(data_path, municipio="VALLEDUPAR")
    n_steps = max(len(serie_valledupar), 50)  # mínimo razonable de pasos

    # 2. Cargar parámetros del modelo SEIR ya calibrado
    beta, sigma, gamma = load_seir_params(params_path)

    # 3. Definir estado inicial y población efectiva
    N = 100_000  # población efectiva aproximada (ajustar si se desea)
    I0 = 1.0
    E0 = 0.0
    R0 = 0.0
    S0 = N - I0 - E0 - R0
    x0 = [S0, E0, I0, R0]

    # 4. Calcular exponente de Lyapunov máximo
    lyap_max = largest_lyapunov(
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        N=N,
        x0=x0,
        n_steps=n_steps,
        dt=1.0,
        renorm_every=5,
    )

    # 5. Guardar resultado en JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "municipio": "Valledupar",
        "lyapunov_max": float(lyap_max),
        "n_steps": int(n_steps),
        "beta": float(beta),
        "sigma": float(sigma),
        "gamma": float(gamma),
        "N": int(N),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Exponente de Lyapunov máximo (Valledupar): {lyap_max:.6f}")
    print(f"Resultado guardado en: {output_path}")


if __name__ == "__main__":
    main()