# Investigación de Cambio Climático y Agua - Análisis Global de Variables Meteorológicas

Este proyecto implementa una investigación integral sobre cambio climático con enfoque específico en sequías, patrones de precipitación y variables atmosféricas críticas. Utilizando datos meteorológicos históricos de múltiples ubicaciones estratégicas del planeta, se analiza el comportamiento del agua en diferentes contextos geográficos y temporales para comprender mejor los impactos, causas y consecuencias futuras del cambio climático.

## 🧠 Descripción del Proyecto

La investigación utiliza **Open-Meteo API** como fuente principal de datos meteorológicos históricos, implementando técnicas de análisis de datos meteorológicos multivariados. A través de la recolección sistemática de datos de ubicaciones clave del planeta (Polo Norte, Polo Sur, Ecuador, Este y Oeste), se construye un dataset comprehensivo para el análisis de patrones climáticos y su evolución temporal.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **API de Datos** | Open-Meteo API | - | Fuente de datos meteorológicos históricos |
| **Análisis de Datos** | Pandas | - | Manipulación de datasets meteorológicos |
| **Análisis Numérico** | NumPy | - | Operaciones numéricas y cálculos estadísticos |
| **Visualización** | Plotly | - | Mapas interactivos y gráficos avanzados |
| **Conectividad** | Requests Cache | - | Optimización de consultas API |
| **Resiliencia** | Retry Requests | - | Manejo robusto de fallos de red |
| **Formato de Datos** | CSV | - | Almacenamiento y intercambio de datos |

## 🌍 Variables Meteorológicas Analizadas

### Variables Principales Utilizadas (Used = True)

| Variable | Unidad | Descripción | Relevancia para Sequías |
|----------|--------|-------------|------------------------|
| `temperature_2m` | °C | Temperatura del aire a 2m sobre el suelo | Factor clave en evapotranspiración |
| `relative_humidity_2m` | % | Humedad relativa a 2m sobre el suelo | Indicador directo de disponibilidad de agua |
| `dew_point_2m` | °C | Temperatura del punto de rocío | Potencial de condensación |
| `precipitation` | mm | Precipitación total (lluvia, nieve) | Variable crítica para sequías |
| `rain` | mm | Precipitación líquida únicamente | Aporte directo de agua |
| `et0_fao_evapotranspiration` | mm | Evapotranspiración de referencia FAO-56 | Pérdida de agua del sistema |
| `vapour_pressure_deficit` | kPa | Déficit de presión de vapor | Estrés hídrico en plantas |

### Variables Complementarias Disponibles

#### Condiciones Atmosféricas
- **Temperaturas aparentes:** Sensación térmica considerando viento y humedad
- **Presión atmosférica:** Presión a nivel del mar y superficie
- **Condiciones de superficie:** Códigos meteorológicos WMO

#### Precipitación y Nieve
- **Nevadas:** Acumulación de nieve en cm
- **Profundidad de nieve:** Manto nival en metros
- **Tipos de precipitación:** Diferenciación entre lluvia y nieve

#### Radiación Solar y Nubes
- **Cobertura nubosa:** Total y por niveles (baja, media, alta)
- **Radiación solar:** Directa, difusa y de onda corta
- **Duración del sol:** Horas efectivas de radiación solar

#### Viento
- **Velocidad del viento:** A 10m y 100m de altura
- **Dirección del viento:** Componentes direccionales
- **Ráfagas:** Velocidades máximas instantáneas

#### Parámetros del Suelo
- **Temperatura del suelo:** A múltiples profundidades (0-7cm, 7-28cm, 28-100cm, 100-255cm)
- **Humedad del suelo:** Contenido volumétrico de agua por capas

## 📄 Pipeline de Investigación

### 1. **Definición de Ubicaciones Estratégicas**

```python
# Ubicaciones clave para análisis global
locations = [
    {"name": "North Pole", "latitude": 90.0, "longitude": 0.0},      # Extremo ártico
    {"name": "South Pole", "latitude": -90.0, "longitude": 0.0},     # Extremo antártico  
    {"name": "Equator", "latitude": 0.0, "longitude": 0.0},          # Zona ecuatorial
    {"name": "East", "latitude": 0.0, "longitude": 90.0},            # Región oriental
    {"name": "West", "latitude": 0.0, "longitude": -90.0},           # Región occidental
]
```

**Justificación de ubicaciones:**
- **Polos:** Extremos climáticos para análisis de variabilidad
- **Ecuador:** Zona de máxima radiación solar y convección
- **Este/Oeste:** Diferentes sistemas meteorológicos y patrones oceánicos
- **Representatividad global:** Cobertura de diferentes zonas climáticas

### 2. **Configuración Robusta de API**

```python
# Sistema resiliente de consultas API
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
```

**Características del sistema:**
- **Cache persistente:** Evita consultas duplicadas y reduce latencia
- **Reintentos automáticos:** 5 intentos con backoff exponencial
- **Optimización de red:** Reutilización de conexiones HTTP
- **Tolerancia a fallos:** Manejo robusto de interrupciones de red

### 3. **Parámetros de Consulta Optimizados**

```python
# Configuración de consulta para datos históricos
params = {
    "latitude": 52.52,                    # Coordenadas específicas por ubicación
    "longitude": 13.41,
    "start_date": "1940-01-01",          # Período histórico extendido
    "end_date": "1940-01-02",
    "hourly": [                          # Variables críticas para sequías
        "temperature_2m",
        "relative_humidity_2m", 
        "dew_point_2m",
        "precipitation",
        "rain",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm", 
        "et0_fao_evapotranspiration",
        "vapor_pressure_deficit"
    ]
}
```

### 4. **Procesamiento Temporal Avanzado**

#### Clasificación de Períodos del Día
```python
def classify_time_of_day(hour):
    """Clasifica horarios en períodos del día para análisis circadiano"""
    if 0 <= hour < 6:
        return "Madrugada"      # Período de mínima evapotranspiración
    elif 6 <= hour < 12:
        return "Mañana"         # Inicio de actividad fotosintética
    elif 12 <= hour < 18:
        return "Tarde"          # Máxima radiación solar y ET
    else:
        return "Noche"          # Descenso de temperatura y humedad
```

**Beneficios del análisis temporal:**
- **Patrones circadianos:** Variaciones diarias de evapotranspiración
- **Picos de estrés hídrico:** Identificación de horas críticas
- **Eficiencia de riego:** Optimización de horarios de irrigación
- **Análisis estacional:** Comparación entre estaciones del año

### 5. **Extracción y Procesamiento de Datos**

```python
# Procesamiento sistemático por ubicación
for location in locations:
    # Configurar coordenadas específicas
    params["latitude"] = location["latitude"]
    params["longitude"] = location["longitude"]
    
    # Consulta a API con manejo de errores
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Extracción de variables hourly
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.to_datetime(hourly.Time(), unit="s", utc=True),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
        "precipitation": hourly.Variables(3).ValuesAsNumpy(),
        "rain": hourly.Variables(4).ValuesAsNumpy(),
        "soil_moisture_0_to_7cm": hourly.Variables(5).ValuesAsNumpy(),
        "soil_moisture_7_to_28cm": hourly.Variables(6).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": hourly.Variables(7).ValuesAsNumpy(),
        "vapor_pressure_deficit": hourly.Variables(8).ValuesAsNumpy(),
    }
```

### 6. **Enriquecimiento de Datos**

```python
# Adición de metadata temporal y geográfica
hourly_dataframe["time_of_day"] = hourly_dataframe["date"].dt.hour.apply(classify_time_of_day)
hourly_dataframe["region"] = location["name"]

# Formateo de fechas para análisis
hourly_dataframe['date'] = hourly_dataframe['date'].dt.strftime('%Y-%m-%d')
```

### 7. **Consolidación y Exportación**

```python
# Unificación de datos de todas las ubicaciones
final_dataframe = pd.concat(all_data, ignore_index=True)

# Exportación para análisis posterior
final_dataframe.to_csv("weather_data_with_regions.csv", index=False)
```

## 🗂️ Estructura del Proyecto

```
📁 climate-water-research/
├── 📄 N1.ipynb                                    # Notebook principal de investigación
├── 📄 N1_export.txt                              # Exportación plana del notebook
├── 📄 weather_data_with_regions.csv              # Dataset consolidado generado
├── 📁 data/
│   ├── 📄 raw/                                   # Datos brutos de API
│   ├── 📄 processed/                             # Datos procesados por región
│   └── 📄 analysis/                              # Resultados de análisis
├── 📁 visualizations/
│   ├── 📄 global_map.html                        # Mapa interactivo de ubicaciones
│   ├── 📄 time_series_plots/                     # Gráficos temporales por variable
│   └── 📄 correlation_matrices/                  # Matrices de correlación
├── 📁 analysis/
│   ├── 📄 drought_indicators.py                  # Cálculo de índices de sequía
│   ├── 📄 trend_analysis.py                      # Análisis de tendencias temporales
│   └── 📄 comparative_analysis.py                # Comparación entre regiones
└── 📄 requirements.txt                           # Dependencias del proyecto
```

## 🚀 Cómo Ejecutar la Investigación

### Instalación de Dependencias

```bash
# Crear entorno virtual
python -m venv climate_research_env
source climate_research_env/bin/activate  # Linux/Mac
# climate_research_env\Scripts\activate  # Windows

# Instalar dependencias
pip install openmeteo-requests
pip install requests-cache retry-requests
pip install pandas numpy plotly
pip install jupyter notebook
```

### Configuración del Entorno

```python
# Verificar instalación de librerías
import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from retry_requests import retry

print("✅ Todas las librerías instaladas correctamente")
```

### Ejecución Paso a Paso

#### Paso 1: Configuración inicial
```python
# Configurar cliente API con cache y reintentos
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

print("🌐 Cliente API configurado exitosamente")
```

#### Paso 2: Definir ubicaciones de estudio
```python
# Ubicaciones estratégicas para análisis global
locations = [
    {"name": "North Pole", "latitude": 90.0, "longitude": 0.0},
    {"name": "South Pole", "latitude": -90.0, "longitude": 0.0},
    {"name": "Equator", "latitude": 0.0, "longitude": 0.0},
    {"name": "East", "latitude": 0.0, "longitude": 90.0},
    {"name": "West", "latitude": 0.0, "longitude": -90.0},
]

print(f"📍 {len(locations)} ubicaciones definidas para análisis")
```

#### Paso 3: Configurar parámetros de consulta
```python
# Parámetros base para consulta API
base_params = {
    "start_date": "1940-01-01",        # Ajustar período según necesidades
    "end_date": "1940-01-02",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m", 
        "precipitation",
        "rain",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "et0_fao_evapotranspiration",
        "vapor_pressure_deficit"
    ]
}
```

#### Paso 4: Recolección de datos
```python
# Ejecutar recolección para todas las ubicaciones
all_data = []
url = "https://archive-api.open-meteo.com/v1/archive"

for i, location in enumerate(locations):
    print(f"📥 Descargando datos para {location['name']} ({i+1}/{len(locations)})")
    
    # Configurar coordenadas específicas
    params = base_params.copy()
    params["latitude"] = location["latitude"]
    params["longitude"] = location["longitude"]
    
    # Realizar consulta
    responses = openmeteo.weather_api(url, params=params)
    # ... procesar respuesta y agregar a all_data
    
print("✅ Descarga de datos completada")
```

#### Paso 5: Procesamiento y análisis
```python
# Consolidar datos de todas las ubicaciones
final_dataframe = pd.concat(all_data, ignore_index=True)

# Análisis exploratorio básico
print("📊 RESUMEN DEL DATASET")
print(f"Total de registros: {len(final_dataframe):,}")
print(f"Rango de fechas: {final_dataframe['date'].min()} a {final_dataframe['date'].max()}")
print(f"Ubicaciones: {final_dataframe['region'].unique()}")
print(f"Variables analizadas: {final_dataframe.select_dtypes(include=[np.number]).columns.tolist()}")

# Guardar resultado
final_dataframe.to_csv("weather_data_with_regions.csv", index=False)
print("💾 Datos guardados en 'weather_data_with_regions.csv'")
```

## 📈 Visualización Interactiva

### Mapa Global de Ubicaciones

El proyecto incluye un mapa interactivo que muestra la distribución de puntos de análisis:

```python
# Crear mapa interactivo con Plotly
fig = go.Figure(go.Scattergeo(
    lon=[loc["longitude"] for loc in locations],
    lat=[loc["latitude"] for loc in locations],
    mode='markers',
    marker=dict(color='red', size=10, symbol='circle'),
    text=[loc["name"] for loc in locations],
    hovertemplate='<b>%{text}</b><br>Lat: %{lat}<br>Lon: %{lon}<extra></extra>'
))

# Configuración de proyección ortográfica
fig.update_geos(
    projection_type="orthographic",
    landcolor='tan',
    oceancolor='deepskyblue',
    showland=True,
    showocean=True,
    bgcolor='lightsteelblue',
    showcountries=True,
    countrycolor="darkgreen",
    coastlinecolor="navy",
)

fig.show()
```

**Características del mapa:**
- **Proyección ortográfica:** Vista esférica realista de la Tierra
- **Marcadores interactivos:** Hover con información de coordenadas
- **Colores personalizados:** Distinción clara entre tierra y océano
- **Escalable:** Fácil adición de nuevas ubicaciones

## 🔬 Aplicaciones de Investigación

### Análisis de Sequías
1. **Índices de sequía:** Cálculo de SPI, PDSI, y índices personalizados
2. **Patrones temporales:** Identificación de ciclos y tendencias
3. **Comparación regional:** Diferencias entre zonas climáticas
4. **Predicción:** Modelos de early warning para sequías

### Estudio del Ciclo del Agua
1. **Balance hídrico:** Precipitación vs evapotranspiración
2. **Eficiencia de agua:** Análisis de pérdidas del sistema
3. **Variabilidad estacional:** Patrones anuales y inter-anuales
4. **Impacto climático:** Efectos del cambio climático en disponibilidad

### Agricultura y Recursos Hídricos
1. **Estrés hídrico en cultivos:** Análisis de VPD y humedad del suelo
2. **Optimización de riego:** Identificación de períodos críticos
3. **Planificación agrícola:** Calendario de siembras basado en clima
4. **Gestión de embalses:** Predicción de aportes hídricos

## 🎯 Objetivos de Investigación

### Objetivos Primarios
1. **Caracterizar patrones globales** de precipitación y evapotranspiración
2. **Identificar tendencias** en variables relacionadas con sequías
3. **Desarrollar indicadores** tempranos de condiciones de sequía
4. **Comparar comportamiento** entre diferentes zonas climáticas

### Objetivos Secundarios
1. **Validar metodologías** de análisis de datos meteorológicos
2. **Establecer líneas base** para estudios futuros
3. **Generar datasets** estandarizados para la comunidad científica
4. **Desarrollar herramientas** de visualización y análisis

### Impacto Esperado
- **Comunidad científica:** Datos y metodologías para investigación climática
- **Gestores de agua:** Herramientas para planificación de recursos hídricos
- **Sector agrícola:** Información para adaptación al cambio climático
- **Políticas públicas:** Evidencia para estrategias de mitigación

## 🔧 Consideraciones Técnicas

### Optimización de Performance
```python
# Gestión eficiente de memoria para grandes datasets
import gc

def optimize_memory_usage(df):
    """Optimizar uso de memoria reduciendo tipos de datos"""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

# Liberación de memoria después de procesamiento
del intermediate_data
gc.collect()
```

### Manejo Robusto de Errores
```python
def robust_api_call(openmeteo, url, params, max_retries=3):
    """Llamada robusta a API con manejo de errores"""
    for attempt in range(max_retries):
        try:
            response = openmeteo.weather_api(url, params=params)
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"⚠️ Error en intento {attempt + 1}, reintentando...")
            time.sleep(2 ** attempt)  # Backoff exponencial
```

### Validación de Datos
```python
def validate_weather_data(df):
    """Validar consistencia de datos meteorológicos"""
    validations = []
    
    # Verificar rangos físicamente posibles
    if (df['temperature_2m'] < -100).any() or (df['temperature_2m'] > 60).any():
        validations.append("❌ Temperaturas fuera de rango físico")
    
    if (df['relative_humidity_2m'] < 0).any() or (df['relative_humidity_2m'] > 100).any():
        validations.append("❌ Humedad relativa fuera de rango 0-100%")
    
    if (df['precipitation'] < 0).any():
        validations.append("❌ Valores negativos en precipitación")
    
    if not validations:
        validations.append("✅ Datos validados correctamente")
    
    return validations
```

## 📊 Métricas y Indicadores Clave

### Indicadores de Sequía
1. **Balance Hídrico:** P - ET₀ (Precipitación - Evapotranspiración)
2. **Índice de Humedad:** RH / VPD ratio
3. **Déficit de Precipitación:** Desviación de medias históricas
4. **Estrés Vegetal:** VPD > 1.6 kPa (umbral crítico)

### Métricas de Calidad de Datos
- **Completitud:** Porcentaje de datos sin valores faltantes
- **Consistencia:** Coherencia entre variables relacionadas
- **Precisión:** Validación contra estándares meteorológicos
- **Actualidad:** Frecuencia de actualización de datos

## 🌱 Líneas de Investigación Futuras

### Expansión Geográfica
1. **Red de estaciones densa:** Aumentar resolución espacial
2. **Cuencas hidrográficas:** Análisis por sistemas hídricos
3. **Gradientes altitudinales:** Efecto de la elevación
4. **Islas climáticas:** Microclimas específicos

### Análisis Avanzados
1. **Machine Learning:** Predicción de sequías con ML
2. **Análisis de frecuencias:** Estadística de extremos
3. **Teleconexiones:** Relación con oscilaciones climáticas
4. **Modelado hidrológico:** Integración con modelos de cuenca

### Integración de Datos
1. **Imágenes satelitales:** NDVI y humedad del suelo desde satélites
2. **Datos de caudales:** Ríos y sistemas hídricos
3. **Información socioeconómica:** Impactos en agricultura
4. **Proyecciones climáticas:** Escenarios futuros de cambio climático

## 📞 Contacto y Colaboración

Para colaboraciones en investigación climática, discusiones sobre metodologías de análisis de sequías, o intercambio de datos meteorológicos, contactar a través de los canales institucionales.

## 📚 Referencias y Recursos

### APIs y Fuentes de Datos
- **Open-Meteo API:** Documentación oficial y mejores prácticas
- **ERA5 Reanalysis:** Datos de reanálisis climático
- **Global Precipitation Climatology:** Bases de datos globales

### Literatura Científica
- **Drought Indices:** Metodologías estándar para análisis de sequías
- **Evapotranspiration:** Modelos FAO-56 Penman-Monteith
- **Climate Change Impacts:** Estudios sobre agua y cambio climático
- **Hydroclimatology:** Fundamentos teóricos del análisis hidro-climático

### Herramientas y Software
- **Python for Climate Science:** Mejores prácticas en análisis climático
- **Pandas Time Series:** Análisis de series temporales meteorológicas
- **Plotly for Geoscience:** Visualización de datos geoespaciales

---

*Esta investigación contribuye al entendimiento global del ciclo del agua y sus alteraciones debido al cambio climático, proporcionando herramientas y datos fundamentales para la gestión sostenible de recursos hídricos y la adaptación a condiciones climáticas cambiantes.*