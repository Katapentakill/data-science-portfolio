# Exploratory Analysis Collection - Geoscience Data Analysis Projects

Esta colección contiene proyectos especializados de análisis exploratorio de datos en geociencias, enfocándose en **cambio climático** y **análisis sísmico**. Cada proyecto implementa técnicas avanzadas de análisis de datos científicos, visualización geoespacial y procesamiento de series temporales para generar insights sobre fenómenos terrestres críticos.

## 🌍 Descripción General

La colección abarca dos dominios fundamentales de las geociencias: **climatología** con enfoque en recursos hídricos y sequías, y **sismología** con análisis de actividad tectónica histórica. Ambos proyectos utilizan metodologías de ciencia de datos para transformar datos geofísicos complejos en visualizaciones interpretables y análisis estadísticos robustos.

## 📊 Tecnologías y Frameworks Utilizados

| Categoría | Tecnologías | Proyectos |
|-----------|-------------|-----------|
| **Lenguajes** | Python 3.x | Ambos proyectos |
| **APIs de Datos** | Open-Meteo API | Change Climate |
| **Análisis de Datos** | Pandas, NumPy | Ambos proyectos |
| **Visualización Estática** | Matplotlib, Seaborn | Ambos proyectos |
| **Visualización Interactiva** | Plotly, Folium | Ambos proyectos |
| **Geoespacial** | GeoPandas, Contextily | Terremoto Visualization |
| **Animaciones** | FuncAnimation (Matplotlib) | Terremoto Visualization |
| **Conectividad** | Requests Cache, Retry Requests | Change Climate |
| **Utilidades** | Tabulate, IPython Display | Ambos proyectos |

## 🏗️ Estructura de la Colección

```
exploratory-analysis/
│
├── Change_Climate/
│   ├── N1.ipynb                                    # Investigación climática principal
│   ├── N1_export.txt                              # Exportación del notebook
│   ├── weather_data_with_regions.csv              # Dataset meteorológico generado
│   └── .cache/                                    # Cache de consultas API
│
├── Terremoto_Visualization/
│   ├── proyecto_terremoto.ipynb                   # Análisis sísmico principal
│   ├── archivo_transformado_log.csv               # Datos sísmicos transformados
│   ├── terremotos_animacion.gif                   # Animación temporal generada
│   └── data/
│       └── Chile Earthquake Dataset (1976-2021).csv  # Dataset sísmico histórico
│
└── README.md                                      # Este archivo de documentación
```

## 🏆 Proyectos de Análisis

### 1. **Change Climate - Investigación de Cambio Climático y Agua** 🌧️
- **Dominio:** Climatología - Análisis de Sequías y Recursos Hídricos
- **Fuente de Datos:** Open-Meteo API (datos históricos meteorológicos)
- **Cobertura Geográfica:** 5 ubicaciones estratégicas globales (Polos, Ecuador, Este, Oeste)
- **Variables Clave:** Precipitación, evapotranspiración, humedad, temperatura, VPD
- **Innovación:** Sistema robusto de consultas API con cache y análisis multirregional

#### Técnicas Implementadas
- **Consultas API optimizadas** con cache persistente y reintentos automáticos
- **Análisis temporal circadiano** con clasificación de períodos del día
- **Procesamiento multirregional** con 5 ubicaciones estratégicas
- **Visualización interactiva** con mapas ortográficos de Plotly

#### Variables Meteorológicas Analizadas
```python
# Variables críticas para análisis de sequías
hourly_variables = [
    "temperature_2m",                    # Temperatura del aire
    "relative_humidity_2m",              # Humedad relativa  
    "dew_point_2m",                     # Punto de rocío
    "precipitation",                     # Precipitación total
    "rain",                             # Precipitación líquida
    "et0_fao_evapotranspiration",       # Evapotranspiración FAO-56
    "vapor_pressure_deficit",           # Déficit de presión de vapor
    "soil_moisture_0_to_7cm",           # Humedad del suelo superficial
    "soil_moisture_7_to_28cm"           # Humedad del suelo profunda
]
```

### 2. **Terremoto Visualization - Análisis Sísmico de Chile (1976-2021)** 🌋
- **Dominio:** Sismología - Análisis Geoespacial de Actividad Tectónica
- **Dataset:** 45 años de registros sísmicos chilenos (1976-2021)
- **Cobertura Geográfica:** Chile continental (latitud -56° a -17°, longitud -80° a -65°)
- **Variables Clave:** Magnitud, profundidad, coordenadas, tensor momento sísmico
- **Innovación:** Animaciones temporales y transformaciones logarítmicas de datos sísmicos

#### Técnicas Implementadas
- **Transformación logarítmica automática** para variables del tensor momento
- **Ingeniería de características temporales** (año, mes, día, hora)
- **Análisis de correlación comprehensivo** con matrices de 20x16 variables
- **Visualización geoespacial multicapa** (estática, animada, interactiva)
- **Animaciones temporales** con FuncAnimation y exportación GIF

#### Pipeline de Transformación de Datos
```python
# Transformación logarítmica inteligente para datos sísmicos
def log_transform(column):
    column_min = column.min()
    if column_min <= 0:
        offset = abs(column_min) + column.max() + 1
        column_adjusted = column + offset
    else:
        column_adjusted = column
    return np.log10(column_adjusted)

# Variables del tensor momento transformadas
log_transformed_columns = ['Mrr', 'Mtt', 'Mpp', 'Mrt', 'Mrp', 'Mtp', 
                          'MrrError', 'MttError', 'MppError', 'MrtError', 
                          'MrpError', 'MtpError', 'moment']
```

## 📈 Resultados y Visualizaciones

### Change Climate - Outputs Principales
| Tipo | Descripción | Formato |
|------|-------------|---------|
| **Dataset Consolidado** | weather_data_with_regions.csv | CSV (5 regiones × variables temporales) |
| **Mapa Interactivo** | Distribución global de ubicaciones | Plotly ortográfico |
| **Series Temporales** | Patrones por región y período del día | Gráficos inline |
| **Cache API** | Consultas optimizadas almacenadas | .cache/ directory |

### Terremoto Visualization - Outputs Principales
| Tipo | Descripción | Formato |
|------|-------------|---------|
| **Matriz de Correlación** | Heatmap 20×16 variables sísmicas | Seaborn inline |
| **Análisis Temporal** | Tendencias anuales de magnitud/profundidad | Matplotlib |
| **Distribuciones** | Frecuencia mensual y diaria de eventos | Gráficos de barras |
| **Mapa de Calor** | Densidad kernel con basemap OSM | GeoPandas + Contextily |
| **Animación Temporal** | Evolución anual 1977-2021 | terremotos_animacion.gif |
| **Mapa Interactivo** | Marcadores por magnitud sísmica | Folium inline |

## 🔬 Metodologías Científicas

### Análisis Climático (Change Climate)

#### Ubicaciones Estratégicas para Representatividad Global
```python
locations = [
    {"name": "North Pole", "latitude": 90.0, "longitude": 0.0},      # Extremo ártico
    {"name": "South Pole", "latitude": -90.0, "longitude": 0.0},     # Extremo antártico  
    {"name": "Equator", "latitude": 0.0, "longitude": 0.0},          # Zona ecuatorial
    {"name": "East", "latitude": 0.0, "longitude": 90.0},            # Región oriental
    {"name": "West", "latitude": 0.0, "longitude": -90.0}            # Región occidental
]
```

#### Sistema de Consultas Resiliente
```python
# Configuración robusta para datos meteorológicos
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
```

**Beneficios metodológicos:**
- **Cache persistente:** Evita consultas redundantes y asegura reproducibilidad
- **Reintentos automáticos:** Robustez ante interrupciones de red
- **Cobertura global:** Representatividad de diferentes zonas climáticas
- **Variables especializadas:** Enfoque específico en indicadores de sequía

### Análisis Sísmico (Terremoto Visualization)

#### Filtrado Geográfico Preciso
```python
# Delimitación exacta de Chile continental
df = df[(df['latitude'] >= -56) & (df['latitude'] <= -17) &
        (df['longitude'] >= -80) & (df['longitude'] <= -65)]
```

#### Transformación de Datos Sísmicos
```python
# Manejo inteligente de valores negativos en tensor momento
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
```

**Beneficios metodológicos:**
- **Normalización de distribución:** Reduce skewness de variables sísmicas
- **Preservación de información:** Mantiene relaciones proporcionales
- **Manejo automático de edge cases:** Valores negativos y ceros
- **Reproducibilidad:** Almacenamiento de offsets aplicados

## 📊 Análisis Estadísticos Implementados

### Técnicas Comunes a Ambos Proyectos

#### Análisis de Correlación
```python
# Matriz de correlación comprehensiva
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
```

#### Ingeniería de Características Temporales
```python
# Extracción sistemática de componentes temporales
def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month  
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    return df
```

#### Análisis de Distribuciones
```python
# Análisis de frecuencias por período temporal
monthly_counts = df['month'].value_counts().sort_index()
yearly_trends = df.groupby('year')[target_variable].agg(['mean', 'min', 'max'])
```

### Técnicas Específicas por Proyecto

#### Change Climate - Análisis Circadiano
```python
def classify_time_of_day(hour):
    if 0 <= hour < 6: return "Madrugada"      # Mínima evapotranspiración
    elif 6 <= hour < 12: return "Mañana"     # Inicio actividad fotosintética  
    elif 12 <= hour < 18: return "Tarde"     # Máxima radiación solar
    else: return "Noche"                     # Descenso T° y humedad
```

#### Terremoto Visualization - Análisis Geoespacial
```python
# Configuración de GeoDataFrame para análisis espacial
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
gdf.set_crs(epsg=4326, inplace=True)  # WGS 84

# Análisis de densidad kernel
sns.kdeplot(data=df, x='longitude', y='latitude', fill=True, 
            cmap='viridis', levels=20, alpha=0.5)
```

## 🚀 Cómo Ejecutar los Proyectos

### Configuración del Entorno
```bash
# Instalación de dependencias base
pip install pandas numpy matplotlib seaborn jupyter

# Para Change Climate
pip install openmeteo-requests requests-cache retry-requests plotly

# Para Terremoto Visualization  
pip install geopandas folium contextily
```

### Ejecución de Change Climate
```python
# 1. Configurar cliente API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. Definir ubicaciones y parámetros
locations = [...]  # 5 ubicaciones estratégicas
params = {...}     # Variables meteorológicas clave

# 3. Ejecutar consultas y procesar datos
# 4. Generar visualizaciones y análisis
```

### Ejecución de Terremoto Visualization
```python
# 1. Cargar dataset sísmico
df = pd.read_csv('Chile Earthquake Dataset (1976-2021).csv', parse_dates=['date'])

# 2. Aplicar filtros geográficos y transformaciones
df = create_date_features(df)
df = apply_log_transforms(df)

# 3. Generar análisis temporal y geoespacial
# 4. Crear animaciones y mapas interactivos
```

## 🎯 Aplicaciones Científicas

### Change Climate - Investigación Climática
- **Análisis de sequías:** Identificación de patrones de déficit hídrico
- **Gestión de recursos hídricos:** Optimización de uso de agua
- **Agricultura sostenible:** Planificación de cultivos según disponibilidad
- **Políticas climáticas:** Evidencia para estrategias de adaptación

### Terremoto Visualization - Sismología Aplicada
- **Evaluación de riesgo sísmico:** Zonificación para construcción
- **Investigación tectónica:** Comprensión de procesos geológicos
- **Gestión de emergencias:** Preparación ante eventos sísmicos
- **Educación pública:** Concientización sobre riesgos naturales

## 📊 Estadísticas de la Colección

### Métricas Generales
- **🌍 2 Dominios Geofísicos:** Climatología y sismología
- **📊 14+ Variables Analizadas:** Meteorológicas y sísmicas
- **⏱️ 80+ Años de Datos:** Desde 1940 (clima) y 1976-2021 (sismos)
- **🗺️ Cobertura Global + Regional:** 5 ubicaciones globales + Chile completo
- **📈 50+ Visualizaciones:** Mapas, series temporales, correlaciones, animaciones

### Distribución por Tipo de Análisis
- **Análisis Temporal:** 40% (series temporales, tendencias, distribuciones)
- **Análisis Geoespacial:** 35% (mapas, densidad kernel, animaciones)
- **Análisis Estadístico:** 20% (correlaciones, transformaciones, agregaciones)
- **Visualización Interactiva:** 5% (Plotly, Folium, widgets)

### Complejidad Técnica
- **Nivel Intermedio:** Change Climate (APIs, cache, análisis multirregional)
- **Nivel Avanzado:** Terremoto Visualization (geoespacial, animaciones, transformaciones)

## 🔧 Consideraciones Técnicas

### Optimización de Performance
```python
# Gestión eficiente de memoria para datasets grandes
import gc

def optimize_dataframe_memory(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# Liberación de memoria
del intermediate_variables
gc.collect()
```

### Reproducibilidad
```python
# Control de aleatoriedad para animaciones y sampling
import random
import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Cache persistente para consultas API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
```

### Manejo de Datos Geoespaciales
```python
# Configuración estándar para proyectos geoespaciales
gdf.set_crs(epsg=4326, inplace=True)  # WGS 84
gdf = gdf.to_crs(epsg=3857)          # Web Mercator para visualización

# Filtros geográficos precisos
chile_bounds = {
    'lat_min': -56, 'lat_max': -17,
    'lon_min': -80, 'lon_max': -65
}
```

## 📞 Contacto y Colaboración

Para consultas sobre metodologías de análisis geofísico, colaboraciones en investigación climática o sísmica, o discusiones sobre técnicas de visualización científica, no dudes en contactar.

## 📗 Referencias y Recursos

### Fuentes de Datos
- **Open-Meteo API:** Datos meteorológicos históricos y en tiempo real
- **Chile Seismic Database:** Registros sísmicos del Centro Sismológico Nacional
- **ERA5 Reanalysis:** Datos climáticos de reanálisis del ECMWF

### Herramientas y Librerías
- **GeoPandas Documentation:** Análisis geoespacial en Python
- **Matplotlib Animation:** Creación de animaciones científicas
- **Plotly Geographic:** Mapas interactivos y proyecciones
- **Seaborn Statistical:** Visualización estadística avanzada

### Literatura Científica
- **Climate Data Analysis:** Metodologías para análisis de variables meteorológicas
- **Seismic Data Processing:** Técnicas de procesamiento de datos sísmicos
- **Geospatial Visualization:** Best practices para visualización geoespacial
- **Time Series Analysis:** Análisis de series temporales en geociencias

---

*Esta colección representa una aplicación comprehensiva de técnicas de análisis exploratorio de datos a fenómenos geofísicos fundamentales, combinando rigor científico con visualización avanzada para generar insights valiosos sobre cambio climático y actividad sísmica que contribuyen al entendimiento y gestión de riesgos naturales.*