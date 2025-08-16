# 🌎 Proyecto Terremotos — Ingeniería de características y análisis anual

Este proyecto realiza **limpieza**, **ingeniería de características** y **análisis anual** sobre un dataset sísmico, generando un CSV normalizado y visualizaciones para entender la **magnitud máxima por año**, la **magnitud inmediatamente anterior** a ese evento y el **recuento de sismos**.

Incluye:
- Un **notebook** (`proyecto_terremoto.ipynb`) para exploración y visualización.
- Un **script** (`guardar.py`) para ejecutar la transformación y guardar resultados de forma reproducible.

> Nota: el pipeline contempla columnas típicas de momento sísmico como `Mrr`, `Mtt`, `Mpp`, `Mrt` y de tiempo (`year`, `month`, `day`, `hour`, `minute`, `second`). Ajusta los nombres si tu dataset difiere.


---

## 🧰 Tecnologías

| Categoría              | Herramienta / Librería |
|-----------------------:|------------------------|
| Lenguaje               | Python 3.9+ |
| Datos                  | pandas, numpy |
| Visualización          | matplotlib, seaborn |
| Tablas en consola      | tabulate |
| Entorno recomendado    | Jupyter Notebook / VS Code + venv |


---

## 📂 Estructura

```
.
├─ guardar.py                   # Script de transformación y guardado
├─ proyecto_terremoto.ipynb     # Notebook de análisis y gráficos
├─ data/                        # (opcional) datasets de entrada
└─ outputs/
   └─ archivo_transformado_normalizado.csv
```

> Si no usas carpetas `data/` y `outputs/`, el script guardará el CSV en el directorio actual.


---

## 🚀 Uso rápido

### 1) Entorno
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip

# Requerimientos mínimos
pip install pandas numpy matplotlib seaborn tabulate
```

### 2) Ejecutar (opción A: Notebook)
1. Abre `proyecto_terremoto.ipynb` en Jupyter/VS Code.
2. Ejecuta todas las celdas en orden.
3. Revisa el CSV generado y las gráficas en la salida del notebook.

### 3) Ejecutar (opción B: Script)
- Abre `guardar.py` y **verifica** las rutas/columnas esperadas.
- Ejecuta:
```bash
python guardar.py
```
- Revisa el archivo `archivo_transformado_normalizado.csv` y los *prints* de resumen en consola.


---

## 🔎 Proceso (pipeline)

1) **Carga y tipificación**
   - Conversión de notación científica y **coerción** a numérico (`errors="coerce"`) para asegurar `float` en todas las columnas cuantitativas.
   - Estandarización de nombres de columnas y verificación de presencia de variables clave (`Mrr`, `Mtt`, `Mpp`, `Mrt`).

2) **Transformación logarítmica segura**
   - Algunas columnas pueden contener valores **negativos o cero** (p. ej., componentes del tensor de momento).
   - Se calcula un **offset** por columna: `offset = abs(min) + max + 1` para desplazar el rango a valores positivos.
   - Se aplica `log10(valor + offset)` y se **registra el offset** usado por columna para permitir la inversión posterior.
   - Se generan nuevas columnas: `Mrr_log`, `Mtt_log`, `Mpp_log`, `Mrt_log`.

3) **Reversión de la transformación** (validación)
   - Se implementa `inverse_log_transform`: `10**(col_log) - offset` para volver a la escala original.
   - Se generan columnas de control como `Mrr_reverted`, etc., para verificar consistencia numérica.

4) **Fecha y orden temporal**
   - A partir de (`year`, `month`, `day`, `hour`, `minute`, `second`) se crea la columna `date` con `pd.to_datetime`.
   - Se ordenan los eventos por `date` para análisis cronológico.

5) **Agregaciones anuales**
   - **Magnitud máxima por año**: `max_magnitude_per_year`.
   - **Magnitud inmediatamente anterior** al evento máximo del año (se ordena por fecha y se toma el registro **anterior** al índice del máximo).
   - **Conteo de sismos por año**: `count`.
   - Se **unen** las tablas por `year` para un *dataset* combinado.

6) **Visualización**
   - Gráfico de líneas para **magnitud máxima** y **magnitud anterior** por `year` (seaborn `lineplot`), con leyenda y *grid*.
   - (Opcional) puedes incorporar el **conteo** como eje secundario o gráficos adicionales (barras/líneas).

7) **Salida**
   - Se guarda `archivo_transformado_normalizado.csv` con las columnas originales + `*_log` + `*_reverted` + columnas temporales.
   - Se imprimen **tablas** de muestra (primeras filas) en consola usando `tabulate` (formato `psql`).


---

## ✅ Resultados esperados

- **CSV enriquecido** con transformaciones y campos temporales.  
- **Gráfica anual** de magnitud máxima vs. anterior (inspección visual de extremos).  
- **Resumen tabular** de las primeras filas para verificación rápida.


---

## ⚠️ Notas y limitaciones

- La métrica “**magnitud anterior al máximo del año**” depende del **orden temporal**. Verifica que la columna `date` esté correctamente construida y que no existan empates/duplicados que alteren el orden.
- La condición `idxmax > 0` en la búsqueda del evento anterior **asume índices consecutivos**: si reseteas índices o tienes índices no enteros, usa posición (`iloc`) tras ordenar por `date`.
- Si faltan columnas (`Mrr`, etc.), **ajusta** la lista `log_transformed_columns` en el script.
- Para evitar pérdidas por `coerce`, revisa proporción de `NaN` tras la conversión numérica.


---

## 🗺️ Roadmap / mejoras sugeridas

- Persistir **offsets** por columna en un archivo JSON para reproducibilidad fuera de memoria.
- Añadir **tests** rápidos (p. ej., confirmar que `reverted ≈ original` dentro de tolerancia).
- Incorporar **análisis de tendencias** (rolling windows) y **detección de outliers**.
- Exportar gráficos a `outputs/figures/` en formatos PNG/SVG.


---

## 📜 Licencia

MIT (o la que definas).

---

### Referencia de código
Parte del pipeline descrito arriba se apoya en utilidades disponibles en los archivos provistos por el usuario. fileciteturn1file0
