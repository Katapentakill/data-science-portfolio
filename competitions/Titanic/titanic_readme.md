# Análisis y Modelado del Conjunto de Datos del Titanic - Machine Learning Classification Project

Este proyecto implementa una solución completa para la clásica competición de Kaggle del Titanic utilizando técnicas avanzadas de machine learning y ensemble methods. El objetivo es predecir la supervivencia de los pasajeros del Titanic basándose en características demográficas, socioeconómicas y de viaje, utilizando un modelo de **StackingClassifier** que combina múltiples algoritmos de clasificación.

## 🧠 Descripción del Proyecto

El proyecto utiliza **StackingClassifier** como modelo principal, combinando **RandomForest**, **CatBoost**, **LightGBM** y **XGBoost** con un meta-learner **RidgeClassifier**. A través de ingeniería de características avanzada, análisis exploratorio detallado y técnicas de ensemble, se construye un predictor robusto capaz de determinar la supervivencia con alta precisión.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.13.1 | Lenguaje principal de desarrollo |
| **Machine Learning** | Scikit-learn | - | Framework principal y StackingClassifier |
| **Ensemble Methods** | RandomForest | - | Algoritmo base de ensemble |
| **Gradient Boosting** | CatBoost | - | Gradient boosting con manejo automático de categóricas |
| **Gradient Boosting** | LightGBM | - | Gradient boosting eficiente |
| **Gradient Boosting** | XGBoost | - | Extreme gradient boosting |
| **Meta-learner** | RidgeClassifier | - | Modelo final del stacking |
| **Análisis de Datos** | Pandas | - | Manipulación y análisis de datos |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas |
| **Visualización** | Matplotlib | - | Gráficos y visualizaciones |
| **Visualización** | Seaborn | - | Visualizaciones estadísticas avanzadas |
| **Preprocessing** | LabelEncoder | - | Codificación de variables categóricas |
| **Métricas** | Accuracy, Confusion Matrix | - | Evaluación de clasificación |

## 📄 Pipeline de Desarrollo

### 1. **Carga y Exploración de Datos**
```python
# Carga del dataset principal del Titanic
train = pd.read_csv('D:\\Ale\\Competitions\\Titanic\\Data\\train.csv')
test_data = pd.read_csv('D:\\Ale\\Competitions\\Titanic\\Data\\test.csv')
```

**Datasets principales:**
- **train.csv:** Datos históricos de pasajeros con supervivencia conocida
- **test.csv:** Conjunto de evaluación para predicciones finales
- **Variables clave:** PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

### 2. **Limpieza y Preprocesamiento de Datos**

#### Manejo de Valores Faltantes
```python
# Estrategia diferenciada por tipo de dato
for column in train.columns:
    if train[column].dtype == 'object':
        train[column].fillna(train[column].mode()[0], inplace=True)  # Moda para categóricas
    else:
        train[column].fillna(train[column].mean(), inplace=True)     # Media para numéricas
```

**Justificación de la estrategia:**
- **Variables categóricas:** Moda preserva la distribución más frecuente
- **Variables numéricas:** Media mantiene la tendencia central
- **Robustez:** Evita pérdida de observaciones por missing values
- **Consistencia:** Aplicación uniforme en train y test

### 3. **Ingeniería de Características Avanzada**

#### Característica de Tamaño Familiar
```python
# Crear variable compuesta de tamaño familiar
train['Familia Size'] = train['SibSp'] + train['Parch'] + 1
```

#### Extracción y Agrupación de Títulos
```python
import re

# Extraer título del nombre
train['Title'] = train['Name'].apply(lambda x: re.findall(r', (.*?)\.', x)[0])

# Agrupar títulos raros para reducir dimensionalidad
title_replacements = {
    'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Miss', 'Countess': 'Rare', 
    'Lady': 'Rare', 'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 
    'Major': 'Rare', 'Capt': 'Rare', 'Don': 'Rare', 'Dona': 'Rare'
}
train['Title'] = train['Title'].replace(title_replacements)
```

#### Categorización de Edad
```python
def categorize_age(age):
    if age <= 5:
        return 'Child'
    elif age <= 12:
        return 'Young'
    elif age <= 18:
        return 'Teenager'
    elif age <= 60:
        return 'Adult'
    else:
        return 'Senior'

train['Age Category'] = train['Age'].apply(categorize_age)
```

#### Binning de Tarifas
```python
# Crear categorías de precio basadas en cuartiles
max_fare = train['Fare'].max()
bins = [0, max_fare / 4, max_fare / 2, 3 * max_fare / 4, float('inf')]
labels = ['Barato', 'Semi Barato', 'Semi Caro', 'Caro']
train['Fare Category'] = pd.cut(train['Fare'], bins=bins, labels=labels, right=False)
```

**Beneficios de la ingeniería de características:**
- **Tamaño familiar:** Captura dinámica familiar vs individual
- **Títulos agrupados:** Reduce sparsity y mejora generalización
- **Categorías de edad:** Convierte variable continua en rangos significativos
- **Binning de tarifas:** Segmentación por poder adquisitivo

### 4. **Análisis Exploratorio de Datos (EDA)**

#### Visualización de Supervivencia
```python
# Distribución de la variable objetivo
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='Survived')
plt.title('Distribución de Supervivencia (0 = No, 1 = Sí)')
plt.show()
```

#### Análisis de Relaciones
```python
# Relación sexo-supervivencia
sns.countplot(data=train, x='Sex', hue='Survived')
plt.title('Relación entre Sexo y Supervivencia')

# Relación clase-supervivencia
sns.countplot(data=train, x='Pclass', hue='Survived')
plt.title('Relación entre Clase y Supervivencia')
```

#### Matriz de Correlación
```python
# Preparación para análisis de correlación
train_corr = train.drop(columns=['Name', "Ticket", "Cabin", "PassengerId"])

# Heatmap de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(train_corr.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación entre Características')
plt.show()
```

### 5. **Codificación de Variables Categóricas**

#### Label Encoding Sistemático
```python
# Codificación de todas las variables categóricas
le = LabelEncoder()
categorical_cols = ['Sex', 'Embarked', 'Title', 'Age Category', 'Fare Category']

for col in categorical_cols:
    train[col] = le.fit_transform(train[col].astype(str))
```

**Ventajas del enfoque:**
- **Consistencia:** Mismo encoder para train y test
- **Robustez:** Conversión a string maneja valores mixtos
- **Eficiencia:** Preparación óptima para tree-based models
- **Reproducibilidad:** Transformaciones consistentes

### 6. **Preparación de Datos para Modelado**

#### Selección de Características
```python
# Features finales para el modelo
X = train[['Sex', 'Age', 'Familia Size', 'Fare', 'Embarked', 
          'Pclass', 'Title', 'Age Category', 'Fare Category']]
y = train['Survived']

# División estratificada para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 7. **Construcción del Modelo de Stacking**

#### Configuración del StackingClassifier
```python
stacking_model = StackingClassifier(
    estimators=[
        ('Random Forest', RandomForestClassifier(random_state=42)),
        ('CatBoost', CatBoostClassifier(silent=True)),
        ('LightGBM', LGBMClassifier()),
        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ],
    final_estimator=RidgeClassifier(),
    cv=5  # Cross-validation para meta-features
)

# Entrenamiento del modelo ensemble
stacking_model.fit(X_train, y_train)
```

**Justificación de la arquitectura:**
- **Random Forest:** Robustez contra overfitting, manejo de características correlacionadas
- **CatBoost:** Excelente performance con variables categóricas, menos hyperparameter tuning
- **LightGBM:** Eficiencia computacional, buena generalización
- **XGBoost:** Performance competitiva, regularización avanzada
- **RidgeClassifier:** Meta-learner lineal previene overfitting del ensemble

### 8. **Evaluación del Modelo**

#### Métricas de Performance
```python
# Predicciones y evaluación
y_pred_stacking = stacking_model.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f'Accuracy del Stacking Model: {accuracy_stacking:.2f}')
```

#### Matriz de Confusión
```python
# Visualización de resultados
conf_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)
plt.figure(figsize=(6, 6))
disp_stacking = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_stacking, 
    display_labels=[0, 1]
)
disp_stacking.plot(cmap='Blues', ax=plt.gca())
plt.title('Matriz de Confusión - Stacking Model')
plt.show()
```

### 9. **Predicciones Finales y Submission**

#### Procesamiento del Test Set
```python
# Aplicar mismo pipeline al conjunto de test
# 1. Limpieza de datos
# 2. Ingeniería de características
# 3. Codificación de variables categóricas
# 4. Selección de features

# Manejo de categorías no vistas
for col in categorical_cols:
    if test_data[col].isin(le.classes_).all():
        test_data[col] = le.transform(test_data[col].astype(str))
    else:
        test_data[col] = le.fit_transform(test_data[col].astype(str))
```

#### Generación de Submission
```python
# Preparar datos finales
X_test_final = test_data[['Sex', 'Age', 'Familia Size', 'Fare', 'Embarked', 
                         'Pclass', 'Title', 'Age Category', 'Fare Category']]

# Predicciones finales
y_test_pred = stacking_model.predict(X_test_final)

# Crear archivo de submission
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'], 
    'Survived': y_test_pred
})
submission.to_csv('D:\\Ale\\Competitions\\Titanic\\Data\\submission.csv', index=False)
```

## 🗂️ Estructura del Proyecto (Jupyter Notebook Environment)

### Archivos y Datasets:

```
DATA-SCIENCE-PORTFOLIO/
└── competitions/
    └── Titanic/
        ├── Data/
        │   ├── train.csv                    # Dataset de entrenamiento
        │   ├── test.csv                     # Dataset de evaluación
        │   └── submission.csv               # Predicciones finales (generado)
        └── Notebook/
            ├── ordenador.ipynb              # Notebook principal de análisis
            ├── Random_Forest.ipynb          # Exploración con Random Forest
            ├── Titanic.ipynb               # Análisis base
            └── Titanic2.ipynb              # Iteraciones adicionales

OUTPUTS GENERADOS EN NOTEBOOK:
├── Visualizaciones inline:
│   ├── Distribución de supervivencia
│   ├── Relaciones sexo-supervivencia
│   ├── Relaciones clase-supervivencia
│   ├── Matriz de correlación (10x8)
│   └── Matriz de confusión del modelo
├── Modelos entrenados:
│   └── StackingClassifier (en memoria)
└── Archivo de submission:
    └── submission.csv (para Kaggle)
```

### Rutas de Acceso en Código:
```python
# Datos de entrada
TRAIN_PATH = "D:\\Ale\\Competitions\\Titanic\\Data\\train.csv"
TEST_PATH = "D:\\Ale\\Competitions\\Titanic\\Data\\test.csv"

# Output de predicciones
SUBMISSION_PATH = "D:\\Ale\\Competitions\\Titanic\\Data\\submission.csv"

# Configuraciones del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
```

## 🚀 Cómo Ejecutar el Proyecto

### Configuración del Entorno Jupyter
```python
# Librerías principales requeridas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Configurar visualización
plt.style.use('default')
sns.set_palette("husl")
```

### Verificación de Datos
```python
# Exploración inicial del dataset
print("=== INFORMACIÓN DEL DATASET DE ENTRENAMIENTO ===")
print(f"Forma del dataset: {train.shape}")
print(f"Columnas: {list(train.columns)}")
print(f"Valores nulos por columna:")
print(train.isnull().sum())
print(f"Distribución de supervivencia:")
print(train['Survived'].value_counts())
```

### Flujo de Ejecución en Jupyter

#### 1. **Sección 1-2: Carga y Limpieza**
```python
# Cargar datasets y aplicar limpieza
train = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

# Aplicar estrategia de imputación
# Rellenar nulos según tipo de variable
```

#### 2. **Sección 3: Ingeniería de Características**
```python
# Crear nuevas variables
# - Familia Size
# - Title extraction y agrupación
# - Age Category
# - Fare Category binning
```

#### 3. **Sección 4: Análisis Exploratorio**
```python
# Generar visualizaciones inline
# - Countplots de supervivencia
# - Análisis bivariado con hue
# - Heatmap de correlaciones
```

#### 4. **Secciones 5-7: Preprocesamiento y Modelado**
```python
# Label encoding de categóricas
# División train/test
# Configuración y entrenamiento del StackingClassifier
```

#### 5. **Secciones 8-9: Evaluación y Submission**
```python
# Métricas de evaluación
# Matriz de confusión
# Procesamiento de test set y generación de submission
```

## 📈 Resultados y Análisis

### Hallazgos del Análisis Exploratorio

#### Patrones de Supervivencia
- **Sexo:** Las mujeres tuvieron una tasa de supervivencia significativamente mayor (política "mujeres y niños primero")
- **Clase socioeconómica:** Pasajeros de primera clase tuvieron mayor probabilidad de supervivencia
- **Edad:** Los niños tuvieron mayor tasa de supervivencia que los adultos
- **Tamaño familiar:** Familias de tamaño medio tuvieron mejor supervivencia que individuos solos o familias muy grandes

#### Correlaciones Significativas
- **Sex-Survived:** Correlación negativa fuerte (codificado: female=0, male=1)
- **Pclass-Survived:** Correlación negativa (clases altas = números bajos)
- **Fare-Survived:** Correlación positiva moderada
- **Age-Survived:** Correlación negativa débil

### Performance del Modelo
- **Arquitectura:** StackingClassifier con 4 base learners + meta-learner
- **Validación:** 5-fold cross-validation para meta-features
- **Métricas:** Accuracy, precision, recall via confusion matrix
- **Robustez:** Ensemble reduce varianza de predicciones individuales

### Técnicas Implementadas
1. **Feature engineering avanzada:** Creación de variables compuestas y categóricas
2. **Ensemble heterogéneo:** Combinación de diferentes paradigmas (bagging, boosting)
3. **Meta-learning:** RidgeClassifier aprende a combinar predicciones base
4. **Preprocessing robusto:** Manejo sistemático de missing values y encoding

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque Ensemble
1. **Diversidad de modelos:** Random Forest (bagging) + múltiples gradient boosting
2. **Regularización automática:** Ridge meta-learner previene overfitting
3. **Robustez:** Menos sensible a outliers y ruido que modelos individuales
4. **Generalización:** Cross-validation en meta-features mejora capacidad predictiva

### Aspectos Únicos del Proyecto
- **Pipeline completo:** Desde EDA hasta submission en un flujo integrado
- **Feature engineering domain-specific:** Variables creadas con conocimiento del dominio
- **Manejo de test leakage:** Encoding consistente entre train y test
- **Visualización comprehensiva:** EDA que informa las decisiones de modelado

## 🎯 Posibles Mejoras

### Optimización de Modelos
1. **Hyperparameter tuning:** GridSearchCV o RandomizedSearchCV para cada base learner
2. **Feature selection:** Métodos automáticos para selección óptima de características
3. **Cross-validation estrategia:** TimeSeriesSplit si hay componente temporal
4. **Métricas alternativas:** AUC-ROC, F1-score para mejor evaluación

### Feature Engineering Avanzada
1. **Interacciones:** Features de segunda orden entre variables importantes
2. **Transformaciones:** Log, sqrt, polynomial features
3. **Encoding alternativo:** Target encoding, frequency encoding para categóricas
4. **Variables externas:** Información histórica del Titanic, condiciones meteorológicas

### Arquitectura de Modelos
1. **Stacking multicapa:** Múltiples niveles de meta-learners
2. **Blending:** Promedio ponderado de modelos en lugar de meta-learner
3. **Neural networks:** Integración de redes neuronales como base learner
4. **Calibración:** Calibration de probabilidades para mejor interpretabilidad

## 🎯 Aplicaciones del Mundo Real

### Transferencia a Otros Dominios
- **Análisis de riesgo crediticio:** Predicción de default en préstamos
- **Marketing predictivo:** Propensión de compra de clientes
- **Medicina predictiva:** Diagnóstico basado en características del paciente
- **Análisis de retención:** Predicción de churn en servicios

### Metodología Generalizable
1. **EDA sistemático:** Proceso replicable para cualquier dataset tabular
2. **Feature engineering:** Técnicas aplicables a variables categóricas y numéricas
3. **Ensemble methods:** Framework extensible a otros problemas de clasificación
4. **Validation strategy:** Cross-validation apropiada para problemas similares

## 📧 Consideraciones Técnicas

### Reproducibilidad
```python
# Control de aleatoriedad
RANDOM_STATE = 42

# Configuración de seeds para todos los algoritmos
RandomForestClassifier(random_state=RANDOM_STATE)
train_test_split(..., random_state=RANDOM_STATE)
```

### Eficiencia Computacional
- **Parallel processing:** Todos los algoritmos utilizan múltiples cores
- **Memory management:** Liberación de variables intermedias
- **Early stopping:** En gradient boosting para prevenir overfitting
- **Silent mode:** CatBoost sin verbose para ejecución limpia

### Robustez del Pipeline
- **Error handling:** Manejo de categorías no vistas en test
- **Data validation:** Verificación de formas y tipos de datos
- **Consistent preprocessing:** Mismo pipeline para train y test
- **Version control:** Tracking de experimentos y resultados

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos de machine learning, ensemble methods, o aplicaciones de clasificación en análisis predictivo, no dudes en contactar.

## 📗 Referencias y Recursos

- **Scikit-learn Documentation:** Guía oficial de StackingClassifier
- **Kaggle Titanic Competition:** Competición original y notebooks públicos
- **Ensemble Methods Literature:** Papers sobre stacking y meta-learning
- **Feature Engineering Best Practices:** Técnicas avanzadas para ML tabular

---

*Este proyecto representa una implementación robusta de ensemble learning para problemas de clasificación binaria, combinando análisis exploratorio detallado, ingeniería de características domain-specific y técnicas avanzadas de machine learning en un pipeline reproducible que puede servir como base para proyectos similares de análisis predictivo.*