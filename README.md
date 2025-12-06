README — Proyecto SEIR–Lyapunov Stability Analyzer (Chagas)
## 1. Título del Proyecto

Análisis de Estabilidad y Dinámica del Modelo SEIR Aplicado a Chagas en Colombia mediante Exponentes de Lyapunov

## 2. Descripción General

Este proyecto desarrolla un sistema integrado para:

- Simular y analizar la dinámica epidemiológica del modelo SEIR ajustado a los casos de Chagas en Colombia, con enfoque particular en la ciudad de Valledupar.

- Calibrar parámetros epidemiológicos, permitiendo explorar distintos escenarios de transmisión.

- Calcular exponentes de Lyapunov, con el propósito de evaluar la estabilidad dinámica del sistema y caracterizar la sensibilidad a las condiciones iniciales.

- Visualizar interactivamente los resultados mediante una aplicación en Streamlit, facilitando el análisis por investigadores, epidemiólogos y tomadores de decisiones.

- El proyecto incorpora buenas prácticas de ingeniería de software y reproducibilidad científica, integrando Git, control de versiones de datos, organización modular de scripts y una aplicación web funcional.

## 3. Estructura del Repositorio

proyecto_lyapunov_chagas/
├── data/
│   ├── seir_simulation.csv               # Simulaciones generadas a partir del modelo SEIR
│   ├── seir_valledupar.csv               # Datos de calibración para Valledupar
│   └── cm_seir_valledupar.png            # Mapas y gráficos utilizados en informes
│
├── models/
│   ├── seir_params.json                  # Parámetros del modelo SEIR
│   ├── seir_params_opt.json              # Parámetros optimizados globales
│   ├── seir_params_opt_valledupar.json   # Parámetros optimizados para Valledupar
│   └── lyapunov_valledupar.json          # Exponente de Lyapunov asociado a la simulación
│
├── notebooks/
│   └── análisis_*                         # Jupyter notebooks para exploración y validación
│
├── src/
│   ├── clean_data.py                      # Limpieza y preparación de datos
│   ├── prepare_data.py                    # Ensambles, transformaciones y particiones
│   ├── simulate_seir.py                   # Implementación del modelo SEIR
│   ├── compute_lyapunov.py                # Cálculo del exponente de Lyapunov
│   └── calibrate_seir.py                  # Ajuste y estimación de parámetros
│
├── app/
│   └── streamlit_app.py                   # Aplicación Streamlit para visualización dinámica
│
├── reports/
│   └── figuras, mapas y simulaciones
│
├── dvc.yaml                               # (Opcional) Esquema para control de datos
├── requirements.txt                       # Dependencias del proyecto
└── README.md

## 4. Pipeline del Proyecto

Aunque este proyecto no implementa un pipeline DVC completo como el anterior, su flujo metodológico puede describirse de la siguiente forma:

### 1. Limpieza y preparación de datos (src/clean_data.py)

- Unificación, verificación y depuración de registros.

- Ajuste de estructuras temporales.

- Normalización y validación de variables epidemiológicas.

### 2. Calibración del modelo SEIR (src/calibrate_seir.py)

- Optimización de parámetros para reproducir tendencias reales.

- Uso de mínimos cuadrados y ajuste por sensibilidad.

### 3. Simulación del modelo SEIR (src/simulate_seir.py)

- Resolución numérica del sistema diferencial.

- Obtención de curvas S, E, I, R, V.

- Identificación de picos, crecimiento inicial y colas epidémicas.

### 4. Cálculo del exponente de Lyapunov (src/compute_lyapunov.py)

- Perturbación de condiciones iniciales.

- Integración sobre la trayectoria del sistema.

- Estimación de la tasa de divergencia:

\[
\lambda = \lim_{t \to \infty} \frac{1}{t} \ln\left( \frac{|\delta(t)|}{|\delta(0)|} \right)
\]

- Interpretación:

    λ < 0 : sistema estable

    λ > 0 : sensibilidad y posible comportamiento caótico

### 5. Visualización interactiva (app/streamlit_app.py)

- Panel para explorar las curvas del modelo.

- Inspección de parámetros calibrados.

- Visualización del exponente de Lyapunov.

- Generación de gráficos dinámicos.

## 5. Modelos Implementados
Modelo principal: SEIR clásico con parametrización epidemiológica. El proyecto utiliza un modelo SEIR extendido, expresado como:

\[
\begin{aligned}
\frac{dS}{dt} &= -\beta S I \\
\frac{dE}{dt} &= \beta S I - \sigma E \\
\frac{dI}{dt} &= \sigma E - \gamma I \\
\frac{dR}{dt} &= \gamma I
\end{aligned}
\]

Con componentes adicionales según la calibración del vector.

***Estabilidad dinámica mediante exponentes de Lyapunov***

Se genera un análisis y caracterización de estabilidad del sistema mediante:

- Perturbaciones pequeñas.

- Comparación de trayectorias.

- Tasa de divergencia acumulada.

## 6. Reproducibilidad
- Instalación del entorno: pip install -r requirements.txt

- Simulación del modelo SEIR: python src/simulate_seir.py

- Cálculo del exponente de Lyapunov: python src/compute_lyapunov.py

- Calibración del modelo: python src/calibrate_seir.py

## 7. Aplicación Streamlit

Este proyecto incluye una aplicación interactiva que permite:

- Visualizar curvas S–E–I–R–V

- Observar picos epidémicos

- Cargar parámetros calibrados

- Visualizar el exponente de Lyapunov

- Explorar escenarios epidemiológicos

- Ejecutar localmente: streamlit run app/streamlit_app.py


## 8. Consideraciones Metodológicas

Las decisiones matemáticas y computacionales se basan en:

- La naturaleza del sistema dinámico SEIR.

- La relevancia del análisis de estabilidad para modelos epidemiológicos.

- La pertinencia de calibrar parámetros específicos para Chagas en Valledupar.

- El uso del exponente de Lyapunov como medida de robustez dinámica.

- La necesidad de reproducibilidad científica mediante scripts modulares.

## 9. Ejecución del Proyecto desde Cero

        git clone https://github.com/ommoraa/proyecto_lyapunov_chagas.git
        cd proyecto_lyapunov_chagas
        pip install -r requirements.txt

        # Simulación y análisis
        python src/simulate_seir.py
        python src/compute_lyapunov.py

## 10. Limitaciones

- La calibración depende de la disponibilidad y calidad de datos reales.

- El modelo SEIR clásico no incorpora:

        heterogeneidad espacial,

        inmunidad parcial,

        mortalidad específica.

- El cálculo del exponente de Lyapunov puede ser sensible a: resolución temporal integración numérica, tamaño de perturbación inicial.

## 11. Líneas Futuras de Trabajo

- Incorporación de modelos vectoriales más complejos.

- Inclusión de mortalidad y natalidad.

- Modelos SEIHRD o SEIRV extendidos.

- Evaluaciones basadas en estabilidad no lineal avanzada.

- Integración con plataformas CI/CD para despliegue automático.

## 12. Licencia: Proyecto con fines académicos y de investigación científica.

## 13. Aplicación Web (Streamlit)

La aplicación web permite explorar los resultados del modelo de manera interactiva:

- curvas epidemiológicas,

- parámetros ajustados,

- valores del exponente de Lyapunov.

Su propósito es facilitar la interpretación y transferencia de resultados hacia comunidades científicas y entidades de salud pública. Acceso en línea: https://proyectolyapunovchagas-vwkyu7flkzouugz9cgg8jr.streamlit.app/

### Nota sobre la elaboración del proyecto
Este proyecto fue elaborado por los autores y contó con asistencia de ChatGPT en la corrección, depuración, estandarización y adaptación del código Python, así como en la organización de notebooks, scripts, documentación técnica y estructuración del pipeline. La responsabilidad final sobre el diseño, decisiones metodológicas y contenido analítico es completamente de los autores.

## 14. Fuentes de Datos

Los datos utilizados para la construcción del modelo SEIR, el ajuste epidemiológico y el cálculo del exponente de Lyapunov provienen del sistema oficial de vigilancia en salud pública de Colombia (SIVIGILA), administrado por el Instituto Nacional de Salud (INS). La información corresponde a los registros del evento Enfermedad de Chagas (Código 205) para el año 2024, obtenidos a través del portal oficial de consulta.

            Instituto Nacional de Salud. (2024). Sistema de Vigilancia en Salud Pública – SIVIGILA: 
            Buscador de eventos (Enfermedad de Chagas). 
            https://portalsivigila.ins.gov.co/Paginas/Buscador.aspx


## Autores  

- Arsenio Hidalgo Troya  

- Oscar Mauricio Mora Arroyo  
  Asignatura: Ciencia de Datos para la Investigación Científica  
  Programa: Doctorado en Ciencias Naturales y Matemáticas  
  Universidad de Nariño (UDENAR) – 2025