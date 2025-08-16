# Predicción de Ventas de Tiendas con CatBoost - Machine Learning Time Series Project

Este proyecto implementa una solución completa para la competición de Kaggle "Store Sales - Time Series Forecasting" utilizando técnicas avanzadas de machine learning y el algoritmo CatBoost. El objetivo es predecir las ventas de productos en diferentes tiendas utilizando datos históricos, información de transacciones, precios del petróleo, días festivos y características de las tiendas.

## 🧠 Descripción del Proyecto

El proyecto utiliza **CatBoost Regressor** como modelo principal para realizar predicciones de series temporales en un contexto de retail. A través de ingeniería de características temporales, fusión de múltiples fuentes de datos y transformaciones logarítmicas, se construye un predictor robusto capaz de estimar las ventas futuras con alta precisión.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Machine Learning** | CatBoost | - | Algoritmo principal de regresión |
| **ML Complementario** | XGBoost | - | Modelo de ensemble (preparado) |
| **ML Tradicional** | Scikit-learn | - | Ridge regression y métricas |
| **Análisis de Datos** | Pandas | - | Manipulación de datasets temporales |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y transformaciones |
| **Visualización** | Matplotlib | - | Gráficos y análisis exploratorio |
| **Visualización** | Seaborn | - | Visualizaciones estadísticas |
| **Preprocessing** | LabelEncoder | - | Codificación de variables categóricas |
| **Métricas** | RMSE | - | Evaluación de regresión |
| **Utilidades** | Tabulate | - | Formateo de resultados |

## 📄 Pipeline de Desarrollo

### 1. **Carga y Exploración de Datos**
```python
# Carga de múltiples datasets relacionados
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
transactions = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')
stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
oil = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/oil.csv')
holidays = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')
test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
```

**Datasets principales:**
- **train.csv:** Datos históricos de ventas por tienda y familia de productos
- **test.csv:** Conjunto de evaluación para predicciones futuras
- **stores.csv:** Información de tiendas (ubicación, tipo, cluster)
- **transactions.csv:** Número de transacciones por tienda y fecha
- **oil.csv:** Precios diarios del petróleo (factor económico)
- **holidays_events.csv:** Días festivos y eventos especiales

### 2. **Preprocesamiento y Ingeniería de Características**

#### Conversión de Fechas y Extracción de Características Temporales
```python
# Conversión a formato datetime
for df in [train, test, transactions, oil, holidays]:
    df['date'] = pd.to_datetime(df['date'])

# Extracción de características temporales
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
```

**Beneficios de la ingeniería temporal:**
- **Captura de patrones estacionales:** Variaciones mensuales y semanales
- **Tendencias anuales:** Cambios de comportamiento año tras año
- **Efectos de día de semana:** Patrones de compra por día
- **Compatibilidad con modelos:** Conversión de fechas a features numéricas

#### Fusión de Datos Multifuente
```python
# Merge complejo con múltiples datasets
train_merged = train.merge(stores, on='store_nbr', how='left') \
                    .merge(transactions, on=['date', 'store_nbr'], how='left') \
                    .merge(oil, on='date', how='left') \
                    .merge(holidays, left_on=['year', 'month', 'day'],
                           right_on=[holidays['date'].dt.year, holidays['date'].dt.month, holidays['date'].dt.day],
                           how='left', suffixes=('', '_holiday'))
```

**Estrategia de fusión:**
- **Left joins:** Preservación de todas las observaciones de ventas
- **Múltiples claves:** Combinación por tienda, fecha y características temporales
- **Gestión de sufijos:** Manejo de columnas duplicadas
- **Contexto enriquecido:** Incorporación de factores externos

### 3. **Limpieza y Transformación de Datos**

#### Eliminación de Características Irrelevantes
```python
# Columnas con alta cardinalidad o información redundante
columns_to_drop = ["transferred", "description", "locale_name", "locale", 
                   "type_holiday", "transactions", "date", "date_holiday"]
train_merged.drop(columns=columns_to_drop, inplace=True, errors='ignore')
test_merged.drop(columns=columns_to_drop, inplace=True, errors='ignore')
```

#### Imputación y Manejo de Valores Faltantes
```python
# Imputación por media para precios del petróleo
train_merged['dcoilwtico'] = train_merged['dcoilwtico'].fillna(train_merged['dcoilwtico'].mean())
test_merged['dcoilwtico'] = test_merged['dcoilwtico'].fillna(test_merged['dcoilwtico'].mean())
```

**Justificación de la estrategia:**
- **Media para variables continuas:** Preservación de la distribución central
- **Robustez:** Manejo de missing values sin pérdida de observaciones
- **Consistencia:** Aplicación uniforme en train y test

### 4. **Codificación de Variables Categóricas**

#### Label Encoding Sistemático
```python
# Codificación de todas las variables categóricas
categorical_cols = data_copy.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_copy[col] = le.fit_transform(data_copy[col].astype(str))
    test_copy[col] = le.transform(test_copy[col].astype(str))
    label_encoders[col] = le
```

**Ventajas del enfoque:**
- **Consistencia entre train/test:** Mismo mapping de categorías
- **Preservación de encoders:** Reutilización para nuevos datos
- **Manejo robusto:** Conversión a string para valores mixtos
- **Compatibilidad:** Preparación para algoritmos tree-based

#### Alineación de Datasets
```python
# Sincronización de columnas entre train y test
data_copy, test_copy = data_copy.align(test_copy, join='left', axis=1, fill_value=0)
```

### 5. **Transformación de la Variable Objetivo**

#### Transformación Logarítmica
```python
# Log-transform para manejar distribución sesgada de ventas
y_log = np.log1p(y)  # np.log1p evita problemas con ventas = 0
```

**Justificación técnica:**
- **Normalización de distribución:** Reduce skewness de ventas
- **Estabilización de varianza:** Mejora homoscedasticidad
- **Manejo de ceros:** log1p(x) = log(1+x) evita log(0)
- **Mejora de performance:** Algoritmos funcionan mejor con datos normalizados

### 6. **División de Datos y Configuración del Modelo**

#### Split Estratificado
```python
# División temporal para validación
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)
```

#### Configuración Optimizada de CatBoost
```python
catboost_model = CatBoostRegressor(
    iterations=100000,          # Capacidad de aprendizaje extendida
    depth=9,                    # Profundidad para capturar interacciones complejas
    learning_rate=0.05,         # Learning rate conservador para estabilidad
    loss_function='RMSE',       # Función de pérdida para regresión
    early_stopping_rounds=100,  # Prevención de overfitting
    verbose=1000                # Monitoreo de progreso
)
```

**Hiperparámetros justificados:**
- **iterations=100000:** Capacidad suficiente para convergencia
- **depth=9:** Balance entre complejidad y generalización
- **learning_rate=0.05:** Velocidad moderada para estabilidad
- **early_stopping=100:** Prevención robusta de overfitting

### 7. **Entrenamiento y Evaluación**

#### Entrenamiento con Validación
```python
# Entrenamiento con conjunto de validación para early stopping
catboost_model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log))
```

#### Evaluación con Transformación Inversa
```python
# Predicciones en escala logarítmica
y_pred_log = catboost_model.predict(X_val)

# Inversión de transformación logarítmica
y_pred = np.expm1(y_pred_log)
y_val = np.expm1(y_val_log)

# Cálculo de RMSE en escala original
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE del modelo CatBoost: {rmse}")
```

**Importancia de la evaluación en escala original:**
- **Interpretabilidad:** RMSE en unidades de ventas reales
- **Validación de transformación:** Verificación de inversión correcta
- **Comparabilidad:** Métrica estándar para competiciones

### 8. **Generación de Predicciones Finales**

#### Predicción en Conjunto de Test
```python
# Predicciones finales con inversión de transformación
y_test_pred_log = catboost_model.predict(test_copy)
y_test_pred = np.expm1(y_test_pred_log)

# Preparación de submission
submission = pd.DataFrame({
    'id': test['id'],
    'sales': y_test_pred
})

submission.to_csv('submission.csv', index=False)
```

## 🏗️ Estructura del Proyecto (Kaggle Environment)

### Entorno de Kaggle - Archivos y Datasets Disponibles:

```
COMPETITIONS:
└── store-sales-time-series-forecasting/
    ├── train.csv                              # Datos históricos de ventas
    ├── test.csv                              # Conjunto de evaluación
    ├── transactions.csv                      # Transacciones por tienda/fecha
    ├── stores.csv                            # Información de tiendas
    ├── oil.csv                               # Precios del petróleo
    ├── holidays_events.csv                   # Días festivos y eventos
    └── sample_submission.csv                 # Formato de envío

NOTEBOOK:
└── store_sales_catboost_forecasting.ipynb   # Notebook principal del proyecto

OUTPUT (/kaggle/working/):
├── catboost_info/                           # Logs y métricas de CatBoost
│   ├── learn/                              # Métricas de entrenamiento
│   └── test/                               # Métricas de validación
└── submission.csv                          # Archivo de envío final
```

### Rutas de Acceso en Código:
```python
# Datos de entrada desde competitions
TRAIN_PATH = "/kaggle/input/store-sales-time-series-forecasting/train.csv"
TEST_PATH = "/kaggle/input/store-sales-time-series-forecasting/test.csv"
TRANSACTIONS_PATH = "/kaggle/input/store-sales-time-series-forecasting/transactions.csv"
STORES_PATH = "/kaggle/input/store-sales-time-series-forecasting/stores.csv"
OIL_PATH = "/kaggle/input/store-sales-time-series-forecasting/oil.csv"
HOLIDAYS_PATH = "/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv"

# Outputs en working directory
SUBMISSION_PATH = "/kaggle/working/submission.csv"
CATBOOST_INFO_PATH = "/kaggle/working/catboost_info/"
```

## 🚀 Cómo Ejecutar el Proyecto en Kaggle

### Configuración del Entorno Kaggle
```python
# Librerías principales disponibles en Kaggle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
```

### Verificación de Datasets
```python
# Verificar disponibilidad de todos los datasets
import os

print("=== COMPETITION DATA ===")
for file in os.listdir('/kaggle/input/store-sales-time-series-forecasting/'):
    print(f"📊 {file}")
    
# Verificar tamaños de datasets
datasets = ['train.csv', 'test.csv', 'transactions.csv', 'stores.csv', 'oil.csv', 'holidays_events.csv']
for dataset in datasets:
    df = pd.read_csv(f'/kaggle/input/store-sales-time-series-forecasting/{dataset}')
    print(f"{dataset}: {df.shape[0]} filas, {df.shape[1]} columnas")
```

### Ejecución Paso a Paso

#### Paso 1: Crear y configurar notebook
1. **Crear notebook** en competición "Store Sales - Time Series Forecasting"
2. **Configurar acelerador**: Settings → Accelerator → GPU P100 (opcional para CatBoost)
3. **Agregar datasets**:
   - **Competition Data**: Store Sales - Time Series Forecasting

#### Paso 2: Análisis exploratorio inicial
```python
# Exploración básica de los datos principales
train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
print("=== INFORMACIÓN DEL DATASET DE ENTRENAMIENTO ===")
print(f"Forma del dataset: {train.shape}")
print(f"Rango de fechas: {train['date'].min()} a {train['date'].max()}")
print(f"Número de tiendas: {train['store_nbr'].nunique()}")
print(f"Número de familias de productos: {train['family'].nunique()}")
print(f"Estadísticas de ventas:")
print(train['sales'].describe())
```

### Flujo de Ejecución en Kaggle

#### 1. **Carga y Preprocessamiento**
```python
# Carga completa de datasets
datasets = {}
file_names = ['train', 'test', 'transactions', 'stores', 'oil', 'holidays_events']
for name in file_names:
    datasets[name] = pd.read_csv(f'/kaggle/input/store-sales-time-series-forecasting/{name}.csv')
    
# Conversión de fechas
for df_name in ['train', 'test', 'transactions', 'oil', 'holidays_events']:
    datasets[df_name]['date'] = pd.to_datetime(datasets[df_name]['date'])
```

#### 2. **Ingeniería de Características**
```python
# Extracción de características temporales
for df_name in ['train', 'test']:
    df = datasets[df_name]
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
```

#### 3. **Fusión de Datos**
```python
# Merge sistemático de todas las fuentes
def merge_all_data(main_df, datasets):
    merged = main_df.copy()
    
    # Merge con stores
    merged = merged.merge(datasets['stores'], on='store_nbr', how='left')
    
    # Merge con transactions
    merged = merged.merge(datasets['transactions'], on=['date', 'store_nbr'], how='left')
    
    # Merge con oil
    merged = merged.merge(datasets['oil'], on='date', how='left')
    
    # Merge con holidays (más complejo por fechas)
    holidays_df = datasets['holidays_events'].copy()
    holidays_df['year'] = holidays_df['date'].dt.year
    holidays_df['month'] = holidays_df['date'].dt.month
    holidays_df['day'] = holidays_df['date'].dt.day
    
    merged = merged.merge(holidays_df[['year', 'month', 'day', 'type', 'locale']], 
                         on=['year', 'month', 'day'], how='left')
    
    return merged

train_merged = merge_all_data(datasets['train'], datasets)
test_merged = merge_all_data(datasets['test'], datasets)
```

#### 4. **Configuración y Entrenamiento de CatBoost**
```python
# Preparación de datos para CatBoost
X = train_processed.drop(['sales', 'id'], axis=1, errors='ignore')
y = train_processed['sales']

# Transformación logarítmica
y_log = np.log1p(y)

# División train/validation
X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42, shuffle=True
)

# Configuración optimizada de CatBoost
model = CatBoostRegressor(
    iterations=100000,
    depth=9,
    learning_rate=0.05,
    loss_function='RMSE',
    early_stopping_rounds=100,
    verbose=1000,
    random_state=42
)

# Entrenamiento con validación
model.fit(X_train, y_train_log, eval_set=(X_val, y_val_log))
```

#### 5. **Evaluación y Predicción Final**
```python
# Evaluación en conjunto de validación
y_pred_log = model.predict(X_val)
y_pred = np.expm1(y_pred_log)
y_val_original = np.expm1(y_val_log)

rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
print(f"RMSE en conjunto de validación: {rmse:.2f}")

# Predicciones finales
X_test = test_processed.drop(['id'], axis=1, errors='ignore')
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)

# Crear submission
submission = pd.DataFrame({
    'id': datasets['test']['id'],
    'sales': test_pred
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Archivo submission.csv creado exitosamente")
```

## 📈 Resultados y Métricas

### Modelo Principal: CatBoost Optimizado
**Configuración final:**
- **Arquitectmo:** Gradient Boosting on Decision Trees
- **Iteraciones:** 100,000 (con early stopping)
- **Profundidad:** 9 niveles de árboles
- **Learning Rate:** 0.05 (conservador para estabilidad)
- **Loss Function:** RMSE (optimizada para regresión)
- **Early Stopping:** 100 rondas sin mejora

### Métricas de Evaluación
- **Métrica principal:** RMSE (Root Mean Squared Error)
- **Transformación:** Evaluación en escala original de ventas
- **Validación:** Hold-out 20% con early stopping
- **Robustez:** Manejo de datos faltantes automático

### Técnicas de Optimización Implementadas
1. **Transformación logarítmica:** Normalización de distribución de ventas
2. **Ingeniería temporal:** Extracción de patrones estacionales
3. **Fusión multifuente:** Incorporación de factores externos
4. **Early stopping:** Prevención automática de overfitting
5. **Imputación inteligente:** Manejo robusto de valores faltantes

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque Time Series
1. **Fusión de datos heterogéneos:** Combinación de ventas, transacciones, economía y calendario
2. **Transformación estabilizadora:** Log-transform para manejo de outliers
3. **Características temporales:** Captura de patrones cíclicos y estacionales
4. **Modelo robusto:** CatBoost maneja automáticamente variables categóricas

### Aspectos Únicos del Proyecto
- **Enfoque multifactor:** Integración de factores económicos (petróleo) y sociales (festivos)
- **Preprocessing automático:** Pipeline robusto para datos del mundo real
- **Escalabilidad:** Manejo eficiente de datasets grandes con CatBoost
- **Robustez temporal:** Consideración de efectos de calendario y estacionalidad

## 🎯 Posibles Mejoras

### Técnicas Avanzadas de Machine Learning
1. **Ensemble de modelos:** Combinación con XGBoost, LightGBM y Random Forest
2. **Stacking avanzado:** Uso de meta-learners para combinación óptima
3. **Feature engineering automático:** Creación de interacciones y polinomios
4. **Optimización de hiperparámetros:** Bayesian optimization o grid search

### Ingeniería de Características Temporales Avanzadas
1. **Lags y windows:** Características de ventas pasadas y promedios móviles
2. **Características de tendencia:** Crecimiento, aceleración y cambios de régimen
3. **Estacionalidad compleja:** Descomposición STL y Fourier features
4. **Eventos especiales:** Modelado específico de Black Friday, Navidad, etc.

### Técnicas de Series Temporales Especializadas
1. **Prophet:** Modelo de Facebook para series temporales con estacionalidad
2. **ARIMA/SARIMA:** Modelos clásicos para componentes autorregresivos
3. **Deep Learning:** LSTM/GRU para patrones temporales complejos
4. **Hybrid models:** Combinación de enfoques estadísticos y ML

### Optimización Avanzada
1. **Cross-validation temporal:** Validación específica para series temporales
2. **Custom loss functions:** Pérdidas asimétricas para retail
3. **Online learning:** Actualización incremental con nuevos datos
4. **Multi-target:** Predicción simultánea de múltiples horizontes

## 🎯 Aplicaciones del Mundo Real

### Impacto en Retail y Supply Chain
- **Planificación de inventario:** Optimización de stock por tienda
- **Gestión de cadena de suministro:** Predicción de demanda para logística
- **Pricing estratégico:** Ajuste de precios basado en demanda predicha
- **Staffing optimization:** Planificación de personal por ventas esperadas

### Escalabilidad y Transferencia
1. **Otros sectores retail:** Aplicación a supermercados, farmacias, e-commerce
2. **Diferentes geografías:** Adaptación a mercados con distintas características
3. **Múltiples horizontes:** Predicción semanal, mensual y trimestral
4. **Real-time deployment:** Sistemas de predicción en tiempo real

## 🔧 Consideraciones Técnicas

### Consideraciones Específicas de Kaggle

#### Limitaciones del Entorno
- **Tiempo de ejecución:** Máximo 12 horas por sesión
- **Memoria RAM:** 16GB disponibles para datasets grandes
- **CPU:** Procesamiento paralelo para CatBoost
- **Almacenamiento:** 20GB en /working para checkpoints

#### Optimizaciones para Kaggle
```python
# Configuración de memoria eficiente para CatBoost
import gc

# Liberación de memoria después de preprocessing
del train_raw, test_raw
gc.collect()

# Configuración de CatBoost para Kaggle
catboost_params = {
    'iterations': 100000,
    'depth': 9,
    'learning_rate': 0.05,
    'thread_count': -1,  # Usar todos los cores disponibles
    'verbose': 1000,
    'allow_writing_files': False  # No escribir archivos temporales
}
```

#### Gestión Eficiente de Datos
```python
# Verificar uso de memoria
def check_memory_usage():
    import psutil
    memory = psutil.virtual_memory()
    print(f"Memoria total: {memory.total / (1024**3):.1f} GB")
    print(f"Memoria disponible: {memory.available / (1024**3):.1f} GB")
    print(f"Memoria usada: {memory.percent:.1f}%")

# Optimización de tipos de datos
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df
```

### Reproducibilidad y Debugging
```python
# Configuración de semillas para reproducibilidad
import random
import os

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Logging detallado para debugging
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_data_info(df, name):
    logger.info(f"Dataset {name}: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(f"Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"Valores nulos: {df.isnull().sum().sum()}")
```

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos de forecasting, o discusiones sobre aplicaciones de machine learning en retail y supply chain, no dudes en contactar.

## 📗 Referencias y Recursos

- **CatBoost Documentation:** Guía oficial y best practices
- **Time Series Forecasting:** Literatura sobre predicción en retail
- **Feature Engineering:** Técnicas avanzadas para series temporales
- **Kaggle Competitions:** Análisis de soluciones ganadoras en forecasting

---

*Este proyecto representa una aplicación integral de machine learning para forecasting en retail, combinando múltiples fuentes de datos, ingeniería de características avanzada y el poder de CatBoost para crear predicciones precisas de ventas que pueden impulsar decisiones estratégicas en el mundo real.*