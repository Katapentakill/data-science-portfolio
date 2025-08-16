# BrisT1D Blood Glucose Prediction - Kaggle Competition Solution

Este proyecto implementa una solución completa para la competición de Kaggle "BrisT1D Blood Glucose Prediction" utilizando técnicas avanzadas de regresión y aprendizaje automático. El objetivo es predecir con precisión los niveles de glucosa en sangre (bg+1:00) basándose en datos históricos y características temporales de pacientes con diabetes tipo 1.

## 🩺 Descripción del Proyecto

El proyecto utiliza el dataset BrisT1D que contiene información de monitoreo continuo de glucosa (CGM) de pacientes con diabetes tipo 1. A través de un proceso exhaustivo de ingeniería de características temporales, preprocesamiento de datos y modelado con XGBoost, se construye un sistema predictivo para ayudar en el manejo de la diabetes.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Análisis de Datos** | Pandas | - | Manipulación y análisis de datos |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y transformaciones |
| **Machine Learning** | Scikit-learn | - | Preprocesamiento y métricas de evaluación |
| **Machine Learning** | XGBoost | - | Algoritmo principal de gradient boosting |
| **Visualización** | Matplotlib | - | Gráficos de dispersión y análisis visual |
| **Utilidades** | Tabulate | - | Formateo de tablas para análisis exploratorio |
| **Preprocesamiento** | LabelEncoder | - | Codificación de variables categóricas |
| **Optimización** | RandomizedSearchCV | - | Optimización de hiperparámetros |

## 🔄 Proceso de Desarrollo

### 1. **Configuración del Entorno**
- Instalación de dependencias: `pandas`, `numpy`, `matplotlib`, `tabulate`, `scikit-learn`, `xgboost`
- Configuración del entorno Kaggle para acceso a datasets

### 2. **Carga y Exploración Inicial**
- Importación de datasets de entrenamiento y prueba desde `/kaggle/input/brist1d/`
- Análisis inicial de la estructura de datos y dimensiones
- Identificación de la variable objetivo: `bg+1:00` (glucosa en sangre en t+1 hora)

### 3. **Limpieza de Datos - Eliminación de Columnas**
**Criterio de eliminación:** Columnas con ≥50% de valores nulos
```python
null_percentage = train.isnull().mean() * 100
columns_to_drop = null_percentage[null_percentage >= 50].index
```

**Impacto:** Reducción significativa de dimensionalidad manteniendo información relevante

### 4. **Ingeniería de Características Temporales**
**Transformación de variable temporal:**
- Descomposición de `time` en: `hours`, `minutes`, `seconds`
- Creación de `total_minutes` para representación continua del tiempo
- Eliminación de componentes redundantes (`seconds`, `time` original)

**Beneficios:**
- Captura de patrones circadianos en los niveles de glucosa
- Representación numérica continua del tiempo para el modelo

### 5. **Análisis Exploratorio de Datos**
- **Visualización temporal:** Gráfico de dispersión `bg+1:00` vs `total_minutes`
- **Identificación de patrones:** Comportamiento de glucosa a lo largo del día
- **Detección de outliers:** Valores extremos en mediciones de glucosa

### 6. **Preprocesamiento Avanzado**

#### Transformación Logarítmica
```python
train_cleaned['log_bg+1:00'] = np.log(train_cleaned['bg+1:00'])
```
**Propósito:** Normalización de la distribución y reducción de sesgo

#### Imputación de Valores Nulos
- **Variables numéricas:** Imputación por media
- **Variables categóricas:** Codificación con LabelEncoder + manejo de valores desconocidos

#### Codificación de Variables Categóricas
- LabelEncoder para todas las columnas de tipo object
- Manejo consistente entre conjuntos de entrenamiento y prueba
- Gestión de categorías no vistas en el conjunto de prueba

### 7. **Modelado con XGBoost**

#### Configuración Base del Modelo
```python
model = XGBRegressor(
    eval_metric='rmse',
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    reg_alpha=0.5,
    reg_lambda=1
)
```

#### Optimización de Hiperparámetros
**Método:** RandomizedSearchCV con 5-fold cross-validation
**Espacio de búsqueda:**
- `learning_rate`: [0.005, 0.01, 0.05, 0.1]
- `max_depth`: [5, 7]
- `n_estimators`: [500, 1000, 1500]
- `reg_alpha`: [0, 0.5, 1]
- `reg_lambda`: [1, 1.5]

**Métrica de optimización:** RMSE (Root Mean Square Error)

### 8. **Evaluación del Modelo**

#### División de Datos
- **Entrenamiento:** 85% del dataset
- **Validación:** 15% del dataset
- **Criterio:** `random_state=42` para reproducibilidad

#### Métricas de Rendimiento
- **RMSE en escala logarítmica:** Evaluación directa del modelo
- **RMSE en escala original:** Transformación inversa con `np.exp()`
- **Análisis de sesgo:** Detección de sesgo sistemático en predicciones

#### Visualización de Resultados
- Gráfico de dispersión: Predicciones vs. Valores Reales
- Línea de referencia perfecta (y = x)
- Análisis visual de la calidad de ajuste

### 9. **Generación de Predicciones Finales**
- Aplicación del modelo optimizado al conjunto de prueba
- Transformación inversa de predicciones logarítmicas
- Creación del archivo `submission.csv` con formato Kaggle

## 📁 Estructura del Proyecto

Este proyecto fue desarrollado completamente en **Kaggle Notebooks**:

```
BrisT1D Blood Glucose Prediction/
│
├── brist1d-glucose-prediction.ipynb    # Notebook principal con análisis completo
├── /kaggle/input/brist1d/
│   ├── train.csv                       # Dataset de entrenamiento
│   ├── test.csv                        # Dataset de prueba
│   └── sample_submission.csv           # Formato de submission
│
└── /kaggle/working/
    └── submission.csv                  # Predicciones finales
```

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Kaggle (Recomendado)
1. Accede al notebook en Kaggle: [BrisT1D Blood Glucose Prediction](kaggle.com/competitions/brist1d)
2. Haz clic en "Copy and Edit" para crear tu propia versión
3. Ejecuta todas las celdas secuencialmente
4. El archivo `submission.csv` se generará automáticamente

### Opción 2: Entorno Local
```bash
# Instalar dependencias
pip install pandas numpy matplotlib tabulate scikit-learn xgboost

# Descargar el dataset desde Kaggle
kaggle competitions download -c brist1d

# Ejecutar el notebook
jupyter notebook brist1d-glucose-prediction.ipynb
```

## 📈 Resultados y Métricas

### Modelo Final: XGBoost Optimizado
**Configuración óptima encontrada por RandomizedSearchCV:**
- Algoritmo base: XGBoost Regressor
- Métrica de entrenamiento: RMSE
- Optimización: 10 iteraciones de búsqueda aleatoria
- Validación: 5-fold cross-validation

**Métricas de Rendimiento:**
- RMSE en conjunto de validación: [valor específico]
- Análisis de sesgo: Detección automática de sesgo positivo/negativo
- Visualización: Gráfico de dispersión predicciones vs. valores reales

### Características del Enfoque Médico
- **Transformación logarítmica:** Manejo apropiado de la naturaleza exponencial de datos glucémicos
- **Ingeniería temporal:** Captura de ritmos circadianos en metabolismo de glucosa
- **Regularización:** Prevención de overfitting crucial en datos médicos
- **Análisis de sesgo:** Importante para aplicaciones de salud

## 🔍 Innovaciones Técnicas

### Fortalezas del Enfoque
1. **Ingeniería de características temporales específica** para datos médicos
2. **Transformación logarítmica apropiada** para variables glucémicas
3. **Optimización sistemática de hiperparámetros** con RandomizedSearchCV
4. **Análisis exhaustivo de sesgo** para validación médica
5. **Visualización especializada** para interpretación clínica

### Consideraciones Médicas
- **Importancia clínica:** Predicción de glucosa crucial para manejo de diabetes
- **Seguridad del paciente:** Análisis de sesgo para evitar predicciones peligrosas
- **Interpretabilidad:** Visualizaciones claras para profesionales de salud
- **Robustez temporal:** Manejo de patrones circadianos y variabilidad glucémica

## 🎯 Posibles Mejoras

### Técnicas Avanzadas
1. **Ensemble methods:** Combinación con Random Forest, LightGBM
2. **Feature engineering temporal avanzado:** Ventanas deslizantes, lag features
3. **Cross-validation temporal:** Validación respetando orden cronológico
4. **Análisis de importancia de características:** SHAP values para interpretabilidad médica

### Consideraciones Clínicas
1. **Validación por subgrupos de pacientes:** Análisis por demografía
2. **Análisis de estabilidad temporal:** Rendimiento en diferentes períodos
3. **Detección de anomalías:** Identificación de valores glucémicos peligrosos
4. **Intervalos de confianza:** Cuantificación de incertidumbre predictiva

## ⚕️ Contexto Médico

### Importancia Clínica
- **Diabetes Tipo 1:** Condición que requiere monitoreo continuo de glucosa
- **Predicción horaria:** Permite ajustes proactivos de insulina
- **Prevención de complicaciones:** Evitar hipo/hiperglucemia severa
- **Calidad de vida:** Mejor control glucémico para pacientes

### Aplicaciones Prácticas
- **Sistemas de alerta temprana:** Notificaciones preventivas
- **Dosificación de insulina:** Optimización de terapia
- **Monitoreo continuo:** Integración con dispositivos CGM
- **Telemedicina:** Seguimiento remoto de pacientes

## 📝 Notas de Implementación

- **Plataforma Kaggle:** Desarrollo optimizado para reproducibilidad en competición
- **Datos sensibles:** Manejo apropiado de información médica anonimizada
- **Escalabilidad:** Código adaptable para datasets de mayor tamaño
- **Reproducibilidad:** `random_state=42` en todos los componentes aleatorios
- **Documentación médica:** Comentarios explicativos para contexto clínico

## 📞 Contacto y Colaboración

Para consultas sobre aspectos técnicos, médicos o colaboraciones en proyectos de salud digital, no dudes en contactar.

---

*Este proyecto fue desarrollado como parte de la competición "BrisT1D Blood Glucose Prediction" en Kaggle, contribuyendo al avance en tecnologías de apoyo para el manejo de diabetes tipo 1.*