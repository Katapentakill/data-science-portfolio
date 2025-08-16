# House Prices Prediction - Kaggle Competition Solution

Este proyecto implementa una solución completa para la competición de Kaggle "House Prices: Advanced Regression Techniques" utilizando técnicas avanzadas de regresión y aprendizaje automático. El objetivo es predecir con precisión los precios de venta de propiedades residenciales basándose en 79 características descriptivas.

## 🏠 Descripción del Proyecto

El proyecto utiliza el dataset "House Prices: Advanced Regression Techniques" de Kaggle, que contiene información detallada sobre propiedades residenciales en Ames, Iowa. A través de un proceso exhaustivo de análisis exploratorio de datos, preprocesamiento y modelado, se construye un sistema predictivo robusto.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Análisis de Datos** | Pandas | - | Manipulación y análisis de datos |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y matrices |
| **Machine Learning** | Scikit-learn | - | Algoritmos de ML y preprocesamiento |
| **Machine Learning** | XGBoost | - | Gradient boosting avanzado |
| **Machine Learning** | LightGBM | - | Gradient boosting eficiente |
| **Machine Learning** | CatBoost | - | Manejo de características categóricas |
| **Visualización** | Matplotlib | - | Gráficos y visualizaciones básicas |
| **Visualización** | Seaborn | - | Visualizaciones estadísticas avanzadas |
| **Interpretabilidad** | SHAP | - | Explicabilidad del modelo |
| **Utilidades** | Tabulate | - | Formateo de tablas para output |

## 🔄 Proceso de Desarrollo

### 1. **Carga y Exploración de Datos**
- Importación de datasets de entrenamiento y prueba
- Análisis inicial de la estructura de datos
- Identificación de tipos de datos (numéricos, categóricos, string)

### 2. **Análisis de Valores Nulos**
- Función personalizada `null_percentage_with_type()` para identificar patrones de valores faltantes
- Análisis del porcentaje de nulos por columna
- Clasificación por tipo de dato para estrategias de imputación diferenciadas

### 3. **Limpieza de Datos - Fase 1**
**Eliminación de columnas con alta proporción de nulos:**
- `PoolQC`, `MiscFeature`, `Alley`, `Fence`, `MasVnrType`, `FireplaceQu`

**Estrategias de imputación:**
- **KNN Imputation** para variables numéricas: `MasVnrArea`, `GarageYrBlt`, `LotFrontage`
- **Imputación por moda** para variables categóricas: características de garaje, sótano y sistema eléctrico

### 4. **Análisis de Variabilidad**
- Función `count_unique_variations()` para analizar la diversidad de valores categóricos
- Identificación de características con poca variabilidad informativa

### 5. **Limpieza de Datos - Fase 2**
**Eliminación de características con baja variabilidad:**
- `Utilities`, `Street`, `LandSlope`, `Condition2`, `RoofMatl`

### 6. **Ingeniería de Características**
- **Codificación de variables categóricas** usando `LabelEncoder`
- **Análisis de correlación** mediante matriz de correlación visual
- **Filtrado por correlación**: eliminación de características con correlación < 0.10 o negativa con la variable objetivo

### 7. **Preparación del Modelo**
- **Transformación logarítmica** de la variable objetivo (`SalePrice`) usando `np.log1p()`
- **División train-test** con proporción 80-20
- **Selección del modelo principal**: CatBoost con hiperparámetros optimizados

### 8. **Modelado y Evaluación**
**Configuración de CatBoost:**
```python
best_params = {
    'learning_rate': 0.03,
    'l2_leaf_reg': 3,
    'iterations': 1000,
    'depth': 4,
    'bagging_temperature': 1,
    'random_state': 42
}
```

**Métrica de evaluación:** RMSE (Root Mean Square Error)

### 9. **Meta-Modelado (Stacking)**
- Implementación de stacking con Ridge Regression como meta-modelo
- Combinación de predicciones de múltiples modelos base
- Mejora de la robustez y precisión predictiva

### 10. **Procesamiento del Conjunto de Prueba**
- Aplicación de las mismas transformaciones al conjunto de prueba
- Manejo de valores nulos específicos del conjunto de test
- Generación de predicciones finales

### 11. **Generación de Resultados**
- Transformación inversa de predicciones (`np.expm1()`)
- Creación del archivo de submission en formato CSV
- Formato compatible con Kaggle: `Id`, `SalePrice`

## 📁 Estructura del Proyecto

Este proyecto fue desarrollado completamente en **Kaggle Notebooks**, por lo que toda la implementación se encuentra en un único notebook:

```
House Prices - Kaggle Competition/
│
├── house-prices-analysis.ipynb     # Notebook principal con todo el análisis
├── /kaggle/input/
│   ├── train.csv                   # Dataset de entrenamiento (1460 filas)
│   ├── test.csv                    # Dataset de prueba (1459 filas)
│   └── data_description.txt        # Descripción de las características
│
└── /kaggle/working/
    └── submission.csv              # Predicciones finales para Kaggle
```

**Estructura del Notebook:**
- **Celdas 1-2**: Importación de librerías y carga de datos
- **Celdas 3-8**: Análisis y limpieza de valores nulos
- **Celdas 9-12**: Eliminación de características con baja variabilidad
- **Celdas 13-15**: Análisis de correlación y filtrado de características
- **Celdas 16-18**: Entrenamiento y evaluación del modelo
- **Celdas 19-25**: Procesamiento del conjunto de prueba y generación de submissions

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Kaggle (Recomendado)
1. Accede al notebook en Kaggle: [House Prices - Advanced Regression Techniques](kaggle.com/code/username/house-prices)
2. Haz clic en "Copy and Edit" para crear tu propia versión
3. Ejecuta todas las celdas secuencialmente
4. El archivo `submission.csv` se generará automáticamente en `/kaggle/working/`

### Opción 2: Entorno Local
```bash
# Instalar dependencias
pip install numpy pandas scikit-learn xgboost lightgbm catboost matplotlib seaborn shap tabulate

# Descargar el dataset desde Kaggle
kaggle competitions download -c house-prices-advanced-regression-techniques

# Ejecutar el notebook
jupyter notebook house_prices_analysis.ipynb
```

**Nota:** El código está optimizado para el entorno Kaggle con rutas específicas (`/kaggle/input/`, `/kaggle/working/`)

## 📈 Resultados y Métricas

El modelo final utiliza un enfoque de stacking que combina:
- **CatBoost** como modelo base principal
- **Ridge Regression** como meta-modelo

**Métricas de rendimiento:**
- RMSE en entrenamiento: [valor específico del modelo]
- RMSE en validación: [valor específico del modelo]

## 🔍 Características Clave del Enfoque

### Fortalezas
1. **Manejo robusto de valores nulos** con estrategias diferenciadas
2. **Selección de características basada en correlación**
3. **Transformaciones apropiadas** (logarítmica para variable objetivo)
4. **Meta-modelado** para mejorar generalización
5. **Visualizaciones comprehensivas** para análisis exploratorio

### Innovaciones Técnicas
- Función personalizada para análisis de nulos con tipado
- Pipeline de limpieza reproducible
- Análisis de variabilidad categórica
- Stacking simplificado pero efectivo

## 🎯 Posibles Mejoras

1. **Validación cruzada** para evaluación más robusta
2. **Optimización de hiperparámetros** automatizada (Grid/Random Search)
3. **Ensemble más diverso** incluyendo modelos lineales y tree-based
4. **Feature engineering avanzado** (interacciones, polinomios)
5. **Manejo de outliers** más sofisticado

## 📝 Notas de Implementación

- **Desarrollado completamente en Kaggle Notebooks** para facilitar la reproducibilidad
- **Dataset integrado**: utiliza directamente los datos de la competición de Kaggle
- **Rutas optimizadas**: código adaptado al sistema de archivos de Kaggle (`/kaggle/input/`, `/kaggle/working/`)
- **Ejecución secuencial**: las celdas deben ejecutarse en orden para mantener dependencias
- **Reproducibilidad garantizada**: `random_state=42` en todos los componentes aleatorios
- **Salida directa**: el archivo de submission se genera automáticamente para envío a Kaggle

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones, no dudes en contactar.

---

*Este proyecto fue desarrollado como parte de la competición "House Prices: Advanced Regression Techniques" en Kaggle.*