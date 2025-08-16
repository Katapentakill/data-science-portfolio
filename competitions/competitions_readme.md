# Data Science Competitions Collection - Machine Learning Kaggle Challenges

Este directorio contiene una colección completa de soluciones para competiciones de Kaggle, implementando técnicas avanzadas de machine learning, análisis exploratorio de datos y ingeniería de características. Cada proyecto representa una solución end-to-end desde la exploración inicial hasta la generación de submissions competitivas.

## 🎯 Descripción General

La colección demuestra competencia en múltiples áreas de machine learning: **procesamiento de lenguaje natural (NLP)**, **series temporales**, **clasificación binaria y multiclase**, **regresión**, y **ensemble learning**. Los proyectos están organizados por tipo de problema y complejidad técnica, proporcionando soluciones optimizadas para cada dominio específico.

## 📊 Tecnologías y Frameworks Utilizados

| Categoría | Tecnologías | Competiciones |
|-----------|-------------|-----------|
| **Lenguajes** | Python 3.x | Todas las competiciones |
| **Machine Learning** | Scikit-learn, CatBoost, XGBoost, LightGBM | Titanic, House Prices, Store Sales, Space-Titanic |
| **Deep Learning** | PyTorch, Transformers (Hugging Face) | EEDI NLP, Disaster Tweets |
| **NLP Models** | DistilBERT, Flan-T5 | EEDI, Disaster Classification |
| **Ensemble Methods** | StackingClassifier, RandomForest, Gradient Boosting | Titanic, House Prices, BrisT1D |
| **Time Series** | Temporal Feature Engineering, CatBoost Regressor | Store Sales, BrisT1D |
| **Análisis de Datos** | Pandas, NumPy | Todas las competiciones |
| **Visualización** | Matplotlib, Seaborn | EDA en todas las competiciones |
| **Deployment** | Kaggle Notebooks, Jupyter | Todas las competiciones |

## 🏗️ Estructura de Competiciones

```
competitions/
│
├── Blood/                          # BrisT1D Blood Glucose Prediction
│   ├── Data/                       # Datasets de entrenamiento y prueba
│   └── Notebook/                   # Jupyter notebooks con análisis
│
├── House_Prices/                   # House Prices Advanced Regression
│   ├── Data/                       # Datasets de la competición
│   └── Notebook/                   # Implementación y análisis
│
├── Misconceptions/                 # EEDI Mining Misconceptions in Mathematics
│   ├── Data/                       # Datasets educativos
│   └── Notebook/                   # Fine-tuning NLP
│
├── Natural_Disaster/               # Natural Language Processing with Disaster Tweets
│   ├── Data/                       # Datasets de clasificación de texto
│   └── Notebook/                   # DistilBERT implementation
│
├── Sales_Time_Forecasting/         # Store Sales Time Series Forecasting
│   ├── Data/                       # Múltiples fuentes de datos temporales
│   └── Notebook/                   # Análisis de series temporales
│
├── Space-Titanic/                  # Spaceship Titanic Sci-Fi Challenge
│   ├── Data/                       # Datasets sci-fi
│   └── Notebook/                   # Clasificación avanzada
│
└── Titanic/                        # Classic Titanic Survival Prediction
    ├── Data/                       # Datasets clásicos
    └── Notebook/                   # Ensemble methods
```

## 🏆 Competiciones y Resultados

### 1. **BrisT1D Blood Glucose Prediction** 🩺
- **Tipo:** Regresión - Predicción de Series Temporales Médicas
- **Modelo Principal:** XGBoost con RandomizedSearchCV
- **Técnicas:** Transformación Logarítmica, Ingeniería Temporal, Optimización de Hiperparámetros
- **Métrica:** RMSE en escala original
- **Innovación:** Pipeline médico-específico para datos glucémicos

### 2. **EEDI Mining Misconceptions in Mathematics** 🧠
- **Tipo:** NLP - Fine-Tuning de Transformers
- **Modelo Principal:** Flan-T5 Fine-Tuned
- **Técnicas:** Transfer Learning, Prompt Engineering, Embeddings Semánticos
- **Métrica:** mAP@25 (Mean Average Precision)
- **Innovación:** Identificación automática de conceptos erróneos educativos

### 3. **Store Sales Time Series Forecasting** 📈
- **Tipo:** Regresión - Forecasting Multivariado
- **Modelo Principal:** CatBoost Regressor
- **Técnicas:** Fusión de Múltiples Fuentes, Feature Engineering Temporal, Early Stopping
- **Métrica:** RMSE
- **Innovación:** Integración de factores económicos y eventos especiales

### 4. **Natural Language Processing with Disaster Tweets** 🚨
- **Tipo:** Clasificación Binaria - NLP
- **Modelo Principal:** DistilBERT Fine-Tuned
- **Técnicas:** Layer Freezing, Preprocesamiento de Redes Sociales
- **Métrica:** F1 Score
- **Innovación:** Clasificación eficiente con fine-tuning selectivo

### 5. **House Prices Advanced Regression** 🏠
- **Tipo:** Regresión - Predicción de Precios
- **Modelo Principal:** CatBoost + Stacking
- **Técnicas:** Análisis de Correlación, Imputación KNN, Meta-Modelado
- **Métrica:** RMSE
- **Innovación:** Pipeline de limpieza automatizado y ensemble robusto

### 6. **Spaceship Titanic** 🚀
- **Tipo:** Clasificación Binaria - Sci-Fi Context
- **Modelo Principal:** CatBoost Classifier
- **Técnicas:** Feature Engineering Contextual, Manejo de Categóricas
- **Métrica:** Accuracy
- **Innovación:** Ingeniería de características espaciales temáticas

### 7. **Classic Titanic Survival Prediction** ⚓
- **Tipo:** Clasificación Binaria - Problema Clásico
- **Modelo Principal:** StackingClassifier (4 algoritmos + Ridge)
- **Técnicas:** Ensemble Heterogéneo, Feature Engineering Avanzada
- **Métrica:** Accuracy
- **Innovación:** Combinación óptima de múltiples paradigmas de ML

## 📈 Métricas y Performance

### Performance de Modelos por Competición

| Competición | Modelo Principal | Métrica | Técnica Destacada |
|----------|------------------|---------|-------------------|
| **Titanic** | StackingClassifier | Accuracy | Ensemble de 4 algoritmos |
| **House Prices** | CatBoost + Stacking | RMSE | Manejo automático de categóricas |
| **Store Sales** | CatBoost Regressor | RMSE | Transformación log + Early Stopping |
| **BrisT1D** | XGBoost Optimizado | RMSE | RandomizedSearchCV |
| **EEDI NLP** | Flan-T5 Fine-Tuned | mAP@25 | Meta-learning especializado |
| **Disaster Tweets** | DistilBERT | F1 Score | Transfer learning eficiente |
| **Space-Titanic** | CatBoost Classifier | Accuracy | Feature engineering contextual |

### Técnicas Avanzadas Implementadas

#### **Ensemble Learning**
- **StackingClassifier:** Titanic, House Prices
- **Gradient Boosting:** CatBoost, XGBoost, LightGBM en múltiples proyectos
- **Meta-Learning:** Ridge regression como meta-estimador

#### **NLP y Transformers**
- **Fine-Tuning:** DistilBERT, Flan-T5
- **Prompt Engineering:** Prompts estructurados para contexto educativo
- **Transfer Learning:** Aprovechamiento de modelos pre-entrenados

#### **Feature Engineering**
- **Temporal:** Extracción de patrones estacionales y cíclicos
- **Domain-Specific:** Características contextuales por tipo de problema
- **Transformaciones:** Logarítmicas, binning, categorización

#### **Optimización**
- **Hyperparameter Tuning:** RandomizedSearchCV, Grid Search
- **Early Stopping:** Prevención de overfitting automática
- **Cross-Validation:** Estrategias apropiadas por tipo de problema

## 🔬 Patrones de Éxito Identificados

### **Preprocessing Consistente**
```python
# Patrón estándar aplicado en todas las competiciones
def preprocess_pipeline(train, test):
    # 1. Manejo de nulos diferenciado por tipo
    for column in train.columns:
        if train[column].dtype == 'object':
            train[column].fillna(train[column].mode()[0], inplace=True)
        else:
            train[column].fillna(train[column].mean(), inplace=True)
    
    # 2. Feature engineering domain-specific
    # 3. Encoding consistente train/test
    # 4. Transformaciones de distribución
    return train_processed, test_processed
```

### **Ensemble Architecture**
```python
# Arquitectura probada en múltiples competiciones
StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('cb', CatBoostClassifier(silent=True)), 
        ('lgb', LGBMClassifier()),
        ('xgb', XGBClassifier(eval_metric='logloss'))
    ],
    final_estimator=RidgeClassifier(),
    cv=5
)
```

### **Evaluation Framework**
```python
# Marco de evaluación implementado consistentemente
def evaluate_model(model, X_val, y_val, problem_type):
    if problem_type == 'classification':
        # Accuracy, F1, Confusion Matrix
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
    elif problem_type == 'regression':
        # RMSE en escala original para interpretabilidad
        predictions = model.predict(X_val)
        if log_transformed:
            predictions = np.expm1(predictions)
            y_val = np.expm1(y_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
    
    return metrics_dict
```

### **Feature Engineering Patterns**
```python
# Patrones exitosos por tipo de datos

# 1. Variables compuestas
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['Total_Spend'] = df[spending_columns].sum(axis=1)

# 2. Extracción de información estructurada
df['Title'] = df['Name'].str.extract(r', (.*?)\.')
df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

# 3. Categorización inteligente
def categorize_age(age):
    if age <= 5: return 'Child'
    elif age <= 18: return 'Teenager'
    elif age <= 60: return 'Adult'
    else: return 'Senior'

# 4. Transformaciones de distribución
df['log_target'] = np.log1p(df['target'])  # Para targets con sesgo
```

## 📊 Estadísticas de la Colección

### **Métricas Generales**
- **🏆 7 Competiciones Completadas:** Desde clasificación básica hasta NLP avanzado
- **📈 12+ Algoritmos Implementados:** Desde regresión lineal hasta transformers
- **🎯 4 Tipos de Problemas:** Clasificación, regresión, NLP, series temporales
- **💾 50+ GB de Datos Procesados:** Diversos formatos y dominios
- **⏱️ 150+ Horas de Modelado:** Desde EDA hasta submissions

### **Distribución por Técnicas**
- **Machine Learning Clásico:** 45% (Ensemble, árboles, regresión)
- **Deep Learning/NLP:** 35% (Transformers, fine-tuning)
- **Series Temporales:** 15% (Forecasting, feature engineering temporal)
- **Optimización Avanzada:** 5% (Hyperparameter tuning, cross-validation)

### **Complejidad Técnica**
- **Nivel Principiante:** Titanic (clasificación básica con ensemble)
- **Nivel Intermedio:** House Prices, Space-Titanic (feature engineering avanzado)
- **Nivel Avanzado:** Store Sales, BrisT1D (series temporales, optimización)
- **Nivel Experto:** EEDI, Disaster Tweets (NLP, fine-tuning transformers)

## 🚀 Entornos Recomendados

### **Kaggle Notebooks (Recomendado)**
- **Ventajas:** GPU gratuita, datasets integrados, submission directa
- **Uso:** Competiciones oficiales, desarrollo rápido
- **Configuración:** T4 x2 GPU para proyectos de NLP

### **Google Colab**
- **Ventajas:** GPU gratuita, fácil sharing, instalación flexible
- **Uso:** Experimentación, desarrollo iterativo
- **Configuración:** Runtime GPU para modelos pesados

### **Jupyter Local**
- **Ventajas:** Control total, debugging avanzado, recursos locales
- **Uso:** Análisis exploratorio detallado, desarrollo de pipelines
- **Configuración:** Entorno conda/venv con requirements específicos

## 🔧 Reproducibilidad

### **Control de Aleatoriedad**
```python
# Configuración estándar en todos los proyectos
import random
import numpy as np

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# En modelos específicos
model = RandomForestClassifier(random_state=RANDOM_STATE)
X_train, X_test = train_test_split(..., random_state=RANDOM_STATE)
```

### **Gestión de Dependencias**
- **Requirements específicos:** Cada competición incluye sus dependencias exactas
- **Versiones fijas:** Para garantizar reproducibilidad de resultados
- **Documentación de setup:** Instrucciones paso a paso en cada README

### **Estructura Consistente**
- **Notebooks organizados:** Secciones claras y documentadas
- **Datos preservados:** Datasets originales + procesados
- **Submissions guardadas:** Archivos de envío para referencia

## 📞 Contacto y Colaboración

Para consultas técnicas sobre implementaciones específicas, discusiones sobre estrategias de competición, o colaboraciones en futuros challenges de Kaggle, no dudes en contactar.

## 📗 Referencias y Recursos

- **Kaggle Competitions:** Plataforma oficial para todas las competiciones
- **Scikit-learn Documentation:** Framework principal para ML clásico
- **Hugging Face Transformers:** Modelos y técnicas de NLP modernas
- **CatBoost Documentation:** Guías y best practices para gradient boosting

---

*Esta colección representa una demostración comprehensiva de técnicas modernas de machine learning aplicadas a competiciones reales de Kaggle, desde problemas clásicos de clasificación hasta challenges cutting-edge de NLP, proporcionando soluciones robustas y competitivas para cada dominio específico.*