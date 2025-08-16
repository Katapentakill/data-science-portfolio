# Análisis Sísmico de Chile - Machine Learning Geoespacial Project

Este proyecto implementa una solución completa para el análisis y visualización de datos sísmicos de Chile (1976-2021) utilizando técnicas avanzadas de ciencia de datos, análisis geoespacial y visualización interactiva. El objetivo es analizar patrones temporales, distribución geográfica y características de los eventos sísmicos para identificar tendencias y crear visualizaciones informativas.

## 🧠 Descripción del Proyecto

El proyecto utiliza **análisis exploratorio de datos**, **ingeniería de características temporales** y **visualización geoespacial** para examinar los patrones sísmicos en Chile a lo largo de 45 años. A través de transformaciones logarítmicas, análisis de correlación y mapas interactivos, se construye un análisis comprehensivo de la actividad sísmica chilena.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Análisis de Datos** | Pandas | - | Manipulación de datasets temporales |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y transformaciones |
| **Visualización** | Matplotlib | - | Gráficos estáticos y animaciones |
| **Visualización** | Seaborn | - | Visualizaciones estadísticas |
| **Geoespacial** | GeoPandas | - | Análisis geoespacial y mapas |
| **Geoespacial** | Folium | - | Mapas interactivos |
| **Geoespacial** | Contextily | - | Mapas base y tiles |
| **Animación** | FuncAnimation | - | Animaciones temporales |
| **Utilidades** | Tabulate | - | Formateo de resultados |

## 📄 Pipeline de Desarrollo

### 1. **Carga y Exploración de Datos**
```python
# Carga del dataset principal de terremotos
file_path = 'Chile Earthquake Dataset (1976-2021).csv'
df = pd.read_csv(file_path, parse_dates=['date'])
```

**Dataset principal:**
- **Chile Earthquake Dataset (1976-2021).csv:** Datos históricos de eventos sísmicos
  - Coordenadas geográficas (latitud, longitud)
  - Información temporal (fecha, hora)
  - Magnitud y profundidad
  - Tensor momento sísmico (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
  - Errores asociados a las mediciones

### 2. **Preprocesamiento e Ingeniería de Características**

#### Filtrado Geográfico
```python
# Filtro para coordenadas de Chile continental
df = df[(df['latitude'] >= -56) & (df['latitude'] <= -17) &
        (df['longitude'] >= -80) & (df['longitude'] <= -65)]
```

#### Extracción de Características Temporales
```python
def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_weekday'] = df['dayofweek'].apply(lambda x: 1 if x < 5 else 0)
    return df

def create_time_features(df):
    df['hour'] = df['time'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['time'].apply(lambda x: int(x.split(':')[1]))
    df['second'] = df['time'].apply(lambda x: int(float(x.split(':')[2])))
    return df
```

**Beneficios de la ingeniería temporal:**
- **Captura de patrones estacionales:** Variaciones mensuales y anuales
- **Análisis de frecuencia:** Distribución por días y horas
- **Detección de tendencias:** Cambios a largo plazo
- **Segmentación temporal:** Análisis por períodos específicos

### 3. **Transformación y Normalización de Datos**

#### Transformación Logarítmica del Tensor Momento
```python
def log_transform(column):
    column_min = column.min()
    
    if column_min <= 0:
        offset = abs(column_min) + column.max() + 1
        column_adjusted = column + offset
        offsets[column.name] = offset
    else:
        column_adjusted = column
        offsets[column.name] = 0
    
    return np.log10(column_adjusted)

# Aplicar transformación a variables del tensor momento
log_transformed_columns = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp', 
                          'MrrError', 'MttError', 'MppError', 'MrtError', 
                          'MrpError', 'MtpError', 'moment']
for col in log_transformed_columns:
    df[col] = log_transform(df[col])
```

**Justificación técnica:**
- **Normalización de distribución:** Reduce skewness de variables sísmicas
- **Estabilización de varianza:** Mejora homoscedasticidad
- **Manejo de valores negativos:** Offset automático para log-transform
- **Preservación de información:** Mantiene relaciones proporcionales

### 4. **Análisis de Correlación**

#### Matriz de Correlación Completa
```python
# Seleccionar solo columnas numéricas
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['event name'])

# Calcular matriz de correlación
correlation_matrix = df_numeric.corr()

# Visualizar como heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap de Matriz de Correlación')
plt.show()
```

### 5. **Análisis Temporal**

#### Tendencias Anuales
```python
# Análisis de magnitud vs año
sns.lmplot(x='year', y='magnitude', data=df, aspect=2, height=6, 
           line_kws={'color': 'orange'})
plt.title('Año vs Magnitud')

# Análisis de profundidad vs año
sns.lmplot(x='year', y='depth', data=df, aspect=2, height=6, 
           line_kws={'color': 'purple'})
plt.title('Año vs Depth')
```

#### Distribución Mensual y Diaria
```python
# Terremotos por mes
monthly_counts = df['month'].value_counts().sort_index()
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='viridis')

# Terremotos por día
daily_counts = df['day'].value_counts().sort_index()
sns.barplot(x=daily_counts.index, y=daily_counts.values, palette='viridis')
```

#### Análisis de Magnitud
```python
# Magnitud promedio por año
average_magnitude_per_year = df.groupby('year')['magnitude'].mean().reset_index()

# Magnitud mínima y máxima por año
min_max_magnitude_per_year = df.groupby('year')['magnitude'].agg(['min', 'max'])
```

### 6. **Visualización Geoespacial**

#### Configuración Geoespacial
```python
# Convertir a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
gdf.set_crs(epsg=4326, inplace=True)  # WGS 84
```

#### Mapa de Calor Estático
```python
# Crear mapa de densidad kernel
fig, ax = plt.subplots(figsize=(15, 30))
sns.kdeplot(data=df, x='longitude', y='latitude', fill=True, 
            cmap='viridis', ax=ax, thresh=0, levels=20, alpha=0.5)

# Añadir basemap de OpenStreetMap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), 
                source=ctx.providers.OpenStreetMap.Mapnik)
```

#### Animación Temporal
```python
def update(year):
    ax.clear()
    yearly_data = df[df['year'] == year]
    
    if not yearly_data.empty:
        sns.kdeplot(data=yearly_data, x='longitude', y='latitude', 
                   fill=True, cmap='viridis', ax=ax, thresh=0, levels=20, alpha=0.5)
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), 
                       source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax.text(0.03, 0.95, f'Año: {year}', transform=ax.transAxes, 
            fontsize=50, fontweight='bold', color='black')

# Crear animación
years = sorted(df['year'].unique())
ani = FuncAnimation(fig, update, frames=years, repeat=True, interval=1000)
ani.save('terremotos_animacion.gif', writer='imagemagick', fps=2)
```

#### Mapa Interactivo
```python
def get_color(magnitude):
    if magnitude > 7:
        return 'red'
    elif magnitude > 4:
        return 'yellow'
    else:
        return 'green'

# Crear mapa Folium
m = folium.Map(location=[-30, -70], zoom_start=5)

# Añadir marcadores por magnitud
for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=5,
        color=get_color(row['magnitude']),
        fill=True,
        fill_color=get_color(row['magnitude']),
        fill_opacity=0.6,
        popup=f'Magnitud: {row["magnitude"]}, Fecha: {row["date"]}'
    ).add_to(m)
```

## 🗂️ Estructura del Proyecto (Jupyter Notebook Environment)

### Archivos y Datasets:

```
PROYECTO_TERREMOTO/
├── data/
│   └── Chile Earthquake Dataset (1976-2021).csv    # Dataset principal
├── proyecto_terremoto.ipynb                        # Notebook principal con análisis
├── archivo_transformado_log.csv                    # Datos transformados (generado)
├── terremotos_animacion.gif                        # Animación temporal (generada inline)
└── README.md                                       # Documentación del proyecto

OUTPUTS GENERADOS EN NOTEBOOK:
├── Gráficos inline:
│   ├── Heatmap de correlación (Cell 4)
│   ├── Análisis temporal año vs magnitud (Cell 5)
│   ├── Distribución mensual/diaria (Cell 6-7)
│   ├── Tendencias temporales (Cell 8-11)
│   └── Mapas geoespaciales (Cell 12)
├── Animación GIF:
│   └── terremotos_animacion.gif (Cell 13-14)
└── Mapa interactivo:
    └── Folium map (Cell 15 - renderizado inline)
```

### Rutas de Acceso en Código:
```python
# Datos de entrada
DATA_PATH = "D:\\Ale\\Competitions\\Data\\Chile Earthquake Dataset (1976-2021).csv"

# Output principal
TRANSFORMED_DATA_PATH = "archivo_transformado_log.csv"
ANIMATION_PATH = "terremotos_animacion.gif"

# Configuraciones geográficas
CHILE_BOUNDS = {
    'lat_min': -56, 'lat_max': -17,
    'lon_min': -80, 'lon_max': -65
}
```

## 🚀 Cómo Ejecutar el Proyecto

### Configuración del Entorno Jupyter
```python
# Configuración para notebook
%matplotlib notebook

# Librerías principales requeridas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import contextily as ctx
from matplotlib.animation import FuncAnimation
from tabulate import tabulate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display

# Configurar estilo de visualización
sns.set(style="whitegrid")
```

### Verificación de Datos
```python
# Verificar estructura del dataset
print("=== INFORMACIÓN DEL DATASET ===")
print(f"Forma del dataset: {df.shape}")
print(f"Rango de fechas: {df['date'].min()} a {df['date'].max()}")
print(f"Rango de latitudes: {df['latitude'].min()} a {df['latitude'].max()}")
print(f"Rango de longitudes: {df['longitude'].min()} a {df['longitude'].max()}")
print(f"Rango de magnitudes: {df['magnitude'].min()} a {df['magnitude'].max()}")
```

### Flujo de Ejecución en Jupyter

#### 1. **Celda 1-3: Carga y Preprocesamiento**
```python
# Cargar dataset y aplicar ingeniería de características
df = pd.read_csv(file_path, parse_dates=['date'])
df = create_date_features(df)
df = create_time_features(df)

# Filtrar por coordenadas de Chile
df = df[(df['latitude'] >= -56) & (df['latitude'] <= -17) &
        (df['longitude'] >= -80) & (df['longitude'] <= -65)]
```

#### 2. **Celda 4: Análisis de Correlación**
```python
# Matriz de correlación y heatmap
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=['event name'])
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
```

#### 3. **Celdas 5-11: Análisis Temporal**
```python
# Series de análisis temporal con visualizaciones inline
# - Tendencias anuales de magnitud y profundidad
# - Distribución mensual y diaria
# - Análisis de magnitudes máximas y mínimas por año
```

#### 4. **Celdas 12-14: Visualización Geoespacial**
```python
# Mapas de calor, animaciones y configuración geoespacial
# Generación de archivo GIF para animación temporal
ani.save('terremotos_animacion.gif', writer='imagemagick', fps=2)
display(Image(filename='terremotos_animacion.gif'))
```

#### 5. **Celda 15: Mapa Interactivo**
```python
# Mapa Folium interactivo renderizado inline
m = folium.Map(location=[-30, -70], zoom_start=5)
# El mapa se visualiza directamente en la celda
```

## 📈 Resultados y Análisis

### Hallazgos Principales del Notebook

#### Distribución Temporal
- **Tendencia anual:** Incremento en la frecuencia de registros sísmicos (visible en gráficos de líneas)
- **Patrones estacionales:** Distribución relativamente uniforme por meses (gráficos de barras)
- **Variabilidad diaria:** Sin patrones claros de concentración por días

#### Características Sísmicas
- **Rango de magnitudes:** 3.0 - 8.8 en escala Richter
- **Profundidades:** Desde superficiales hasta 200+ km
- **Distribución geográfica:** Concentración en la cordillera y zona de subducción (mapas de calor)

#### Correlaciones Significativas (Heatmap Cell 4)
- **Tensor momento:** Correlaciones esperadas entre componentes
- **Magnitud-momento:** Correlación positiva fuerte
- **Errores de medición:** Correlacionados con la complejidad del evento

### Visualizaciones Generadas en el Notebook
1. **Heatmap de correlación (Cell 4):** Matriz 20x16 con todas las variables
2. **Gráficos temporales (Cells 5-11):** Tendencias y distribuciones
3. **Mapa de calor geográfico (Cell 12):** Densidad kernel con basemap OSM
4. **Animación temporal (Cells 13-14):** GIF mostrando evolución por años
5. **Mapa interactivo (Cell 15):** Folium con marcadores por magnitud

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque en Jupyter
1. **Análisis iterativo:** Desarrollo paso a paso con visualización inmediata
2. **Transformación automática:** Manejo inteligente de valores negativos en log-transform
3. **Integración multi-temporal:** Análisis desde escalas horarias hasta décadas
4. **Visualización inline:** Resultados inmediatos sin archivos externos

### Aspectos Únicos del Proyecto
- **Pipeline completo en notebook:** Desde carga hasta visualización avanzada
- **Transformaciones robustas:** Preprocessing automatizado para datos sísmicos complejos
- **Múltiples tipos de visualización:** Estática, animada e interactiva en un solo flujo
- **Análisis geoespacial integrado:** Combinación de estadística y cartografía

## 🎯 Posibles Mejoras

### Extensiones del Notebook
1. **Celdas adicionales de ML:** Clustering espacial y clasificación de eventos
2. **Dashboard interactivo:** Widgets para filtros dinámicos
3. **Análisis de series temporales:** Modelos ARIMA para predicción
4. **Validación estadística:** Tests de significancia para tendencias

### Optimización del Workflow
1. **Funciones modulares:** Refactorización en módulos Python
2. **Parámetros configurables:** Variables globales para ajustes rápidos
3. **Cache de resultados:** Almacenamiento de cálculos costosos
4. **Exportación automática:** Scripts para generar reportes

### Visualización Avanzada
1. **Plotly interactivo:** Gráficos web más dinámicos
2. **Mapas 3D:** Visualización de profundidad
3. **Dashboard Streamlit:** Aplicación web desde el notebook
4. **Realidad aumentada:** Integración con herramientas AR

## 🎯 Aplicaciones del Mundo Real

### Impacto en Gestión de Riesgos
- **Planificación urbana:** Zonificación sísmica basada en análisis histórico
- **Ingeniería estructural:** Especificaciones de diseño antisísmico
- **Gestión de emergencias:** Preparación basada en patrones identificados
- **Seguros y finanzas:** Evaluación de riesgo para pólizas

### Transferencia y Escalabilidad
1. **Otros países sísmicos:** Adaptación del notebook a diferentes regiones
2. **Monitoreo en tiempo real:** Extensión con feeds de datos en vivo
3. **Educación:** Uso como material didáctico en sismología
4. **Investigación:** Base para estudios científicos especializados

## 📧 Consideraciones Técnicas

### Optimización para Jupyter
- **Gestión de memoria:** Liberación de variables grandes entre celdas
- **Tiempos de ejecución:** Optimización de loops y cálculos matriciales
- **Reproducibilidad:** Seeds fijos y control de versiones de librerías
- **Documentación inline:** Markdown descriptivo entre celdas de código

### Dependencias y Compatibilidad
```python
# Versiones recomendadas para reproducibilidad
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
geopandas>=0.9.0
folium>=0.12.0
contextily>=1.1.0
```

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos de sismología, análisis geoespacial en Jupyter, o aplicaciones de ciencia de datos en geociencias, no dudes en contactar.

## 📗 Referencias y Recursos

- **Jupyter Documentation:** Best practices para análisis científico
- **GeoPandas Tutorials:** Análisis geoespacial en notebooks
- **Seismology Python Libraries:** ObsPy y herramientas especializadas
- **Chile Seismic Networks:** CSN (Centro Sismológico Nacional)

---

*Este proyecto representa una aplicación integral de ciencia de datos en Jupyter Notebook para el análisis de fenómenos sísmicos, combinando análisis estadístico, ingeniería de características temporales y visualización geoespacial avanzada en un flujo interactivo que facilita la exploración y comprensión de la actividad sísmica de Chile.*