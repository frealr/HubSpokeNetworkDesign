# 8-node Spain Hub-and-Spoke Network

Red de 8 aeropuertos españoles: MAD, BCN, PMI, AGP, ALC, LPA, TFS, IBZ.

## Estructura del directorio

```
8node_spain/
├── export_params.py          # Escribe export_txt/ sin lanzar GAMS (punto de entrada recomendado)
├── generate_data.py          # Genera parámetros y lanza las simulaciones completas
├── compare_results.py        # Compara resultados MIP vs BLO y genera gráficas
├── param_definition.gms      # Definición de parámetros para GAMS (incluye export_txt/)
├── mip.gms                   # Modelo MIP en GAMS
├── cvx-ll.gms                # Subproblema de localización (BLO)
├── cvx-sl.gms                # Subproblema de flujos (BLO)
├── requirements.txt          # Dependencias Python
├── distance.csv              # Matriz de distancias entre nodos (km)
├── demand.csv                # Matriz de demanda diaria entre pares O-D (pasajeros)
├── prices.csv                # Matriz de precios de billete entre pares O-D
├── export_txt/               # Parámetros en formato texto para GAMS
├── 8node_hs_prueba_v0/       # Resultados MIP (.mat por combinación budget/lam)
└── 8node_hs_prueba_v0_blo/   # Resultados BLO (.mat por combinación de parámetros)
```

## Requisitos

- Python ≥ 3.9
- GAMS ≥ 49.6 instalado en el sistema

Instalar dependencias Python:

```bash
pip install -r requirements.txt
```

## Parámetros configurables

Todos los parámetros de experimentación están al final de `generate_data.py`,
en el bloque `if __name__ == '__main__'`. **No es necesario tocar el modelo**
(`.gms`) para variar estos valores.

| Parámetro | Variable en el script | Valor por defecto | Descripción |
|---|---|---|---|
| Presupuestos | `budgets` | `[3e4, 4e4, …, 1e5]` | Lista de presupuestos a evaluar (€/día) |
| Lambda (coste de hub) | `lam` | `4` | Multiplicador del coste de apertura de hub respecto al nodo |
| Alfa (peso pasajeros) | `alfas` | `[0.1]` | Peso del término de demanda en el objetivo BLO |
| Iters externas BLO | `niters` (en `compute_sim_cvx_blo`) | `20` | Número de iteraciones del bucle externo del método BLO |
| Iters bloque BLO | `bliters` (en `compute_sim_cvx_blo`) | `30` | Número máximo de iteraciones por bloque |
| Paso gradiente alfa | `mu_alfa` (en `compute_sim_cvx_blo`) | `1e-7` | Tasa de aprendizaje para el multiplicador alfa |
| Paso gradiente beta | `mu_beta` (en `compute_sim_cvx_blo`) | `2e-1` | Tasa de aprendizaje para el multiplicador beta |
| Semilla aleatoria | `np.random.seed(123)` (en `parameters_8node_network`) | `123` | Semilla para reproducibilidad de utilidades alternativas |

### Parámetros físicos de la red

Definidos dentro de `parameters_8node_network()` en `generate_data.py`:

| Parámetro | Valor | Descripción |
|---|---|---|
| `omega_t` | `-0.02` | Sensibilidad al tiempo de viaje en la utilidad logit |
| `omega_p` | `-0.02` | Sensibilidad al precio en la utilidad logit |
| `link_cost` | `10 × distancia` | Coste de construcción por enlace (€/km) |
| `station_cost` | `3 000 €` por nodo | Coste fijo de apertura de nodo |
| `hub_cost` | `5 000 €` por nodo | Coste fijo adicional por abrir como hub |
| `a_nom` | `171` | Capacidad nominal de aeronave (asientos) |
| `tau` | `0.85` | Factor de llenado máximo de aeronave |
| `eta` (sigma) | `0.3` | Elasticidad de costes operativos |
| `n_airlines` | `5` | Número de aerolíneas alternativas en el modelo logit |

## Cómo generar los datos y lanzar las simulaciones

### 0. Solo generar los parámetros (sin ejecutar GAMS)

```bash
cd 8node_spain
python3 export_params.py --budget 40000 --lam 4 --alfa 0.1
```

Escribe todos los archivos de `export_txt/` que GAMS lee vía `$include`. No lanza ningún
subproceso. Tras ejecutarlo, se puede llamar a GAMS directamente sobre cualquiera de los
tres modelos (`mip.gms`, `cvx-ll.gms`, `cvx-sl.gms`).

Opciones disponibles:

```
--budget    FLOAT   Presupuesto diario en €/día (obligatorio)
--lam       FLOAT   Multiplicador coste hub (default: 4)
--alfa      FLOAT   Peso demanda en objetivo BLO (default: 0.1)
--dm-pax    FLOAT   Pendiente demanda pasajeros en MIP (default: 0.01)
--dm-op     FLOAT   Pendiente costes operativos en MIP (default: 0.008)
--gamma     FLOAT   Parámetro gamma del método BLO (default: 20)
--niters    INT     Iteraciones externas BLO (default: 20)
```

### 1. Generar parámetros y ejecutar ambos métodos (MIP + BLO)

```bash
cd 8node_spain
python generate_data.py
```

El script:
1. Calcula todos los parámetros de la red a partir de `distance.csv`, `demand.csv` y `prices.csv`.
2. Escribe los archivos de texto en `export_txt/` (leídos por GAMS vía `$include`).
3. Lanza GAMS con `mip.gms` para cada combinación `(budget, lam)` → guarda en `8node_hs_prueba_v0/`.
4. Lanza el método BLO con multistart para cada combinación `(budget, alfa)` → guarda en `8node_hs_prueba_v0_blo/`.

Para cambiar los presupuestos o lambda, edita estas líneas en `generate_data.py`:

```python
# --- PARÁMETROS DE EXPERIMENTACIÓN ---
budgets = [3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]
lam     = 4
alfas   = [0.1]
```

### 2. Indicar la ruta al ejecutable de GAMS

Por defecto se usa `/opt/gams/gams49.6_linux_x64_64_sfx/gams`. Para usar otra instalación:

```bash
export GAMS_EXE=/ruta/a/tu/gams
python generate_data.py
```

### 3. Comparar resultados y generar gráficas

```bash
python compare_results.py
```

Opciones disponibles:

```
--budgets   1e4 2e4 ...    Presupuestos a comparar (por defecto: 3e4 4e4 5e4 6e4 7e4 8e4 9e4 1e5)
--lam       4              Valor de lambda
--alfa      0.1            Valor de alfa
--mu-alfa   1e-7           mu_alfa usado al generar los resultados BLO
--mu-beta   0.2            mu_beta usado al generar los resultados BLO
--output    figura.png     Ruta de la figura de gap de optimalidad
--time-output figura.png   Ruta de la figura de tiempos de cómputo
```

Ejemplo con parámetros distintos:

```bash
python compare_results.py --budgets 3e4 5e4 8e4 --lam 4 --alfa 0.1 --mu-alfa 1e-5 --mu-beta 0.05
```

## Formato de los archivos de resultados

Cada ejecución genera un `.mat` en `8node_hs_prueba_v0_blo/` con el nombre:

```
bud=<budget>_lam=<lam>_alfa=<alfa>_mu_al=<mu_alfa>_mu_bet=<mu_beta>_python-euler.mat
```

Contenido del `.mat`:

| Variable | Descripción |
|---|---|
| `s` | Vector de capacidades de nodo abierto |
| `sh` | Vector de capacidades extra de hub |
| `a` | Matriz de frecuencias de vuelo por enlace |
| `f` | Matriz de fracción de demanda atendida por el operador |
| `fext` | Matriz de fracción de demanda atendida por la competencia |
| `fij` | Tensor de flujos por segmento `(i,j,o,d)` |
| `comp_time` | Tiempo total de cómputo (s) |
| `pax_obj` | Ingresos totales por pasajero |
| `op_obj` | Costes operativos totales |
| `obj_val` | Valor del objetivo (ingresos − costes) |
| `alfa_od`, `beta_od` | Multiplicadores finales del método BLO |
| `s_traj`, `sh_traj`, `f_traj` | Trayectorias a lo largo de las iteraciones |
