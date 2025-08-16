# Spaceship Titanic Prediction - Sci-Fi Machine Learning Challenge

Este proyecto implementa una solución completa para la competición de Kaggle "Spaceship Titanic" utilizando técnicas avanzadas de ingeniería de características y algoritmos de machine learning. El objetivo es predecir qué pasajeros de una nave espacial fueron transportados a una dimensión alternativa durante un encuentro con una anomalía espaciotemporal.

## 🚀 Descripción del Proyecto

El proyecto utiliza el dataset futurista "Spaceship Titanic" que contiene información detallada sobre 8,693 pasajeros de una nave espacial de lujo. A través de análisis exploratorio, ingeniería de características específicas para el contexto espacial y modelado con CatBoost, se construye un sistema predictivo para determinar el destino de los pasajeros durante el viaje intergaláctico.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Análisis de Datos** | Pandas | - | Manipulación y análisis de datasets espaciales |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y transformaciones |
| **Visualización** | Matplotlib | - | Gráficos y visualizaciones de datos |
| **Visualización** | Seaborn | - | Visualizaciones estadísticas avanzadas |
| **Machine Learning** | Scikit-learn | - | Preprocesamiento y métricas de evaluación |
| **Gradient Boosting** | CatBoost | - | Algoritmo principal especializado en categóricas |
| **Ensemble Methods** | RandomForestClassifier | - | Modelo de árboles de decisión |
| **Gradient Boosting** | XGBoost | - | Gradient boosting optimizado |
| **Gradient Boosting** | LightGBM | - | Gradient boosting eficiente |
| **Meta-Learning** | StackingClassifier | - | Ensamblado de múltiples modelos |
| **Regularización** | RidgeClassifier | - | Meta-estimador para stacking |
| **Preprocesamiento** | LabelEncoder | - | Codificación de variables categóricas |
| **Utilidades** | Tabulate | - | Formateo de tablas para análisis |

## 📄 Proceso de Desarrollo

### 1. **Carga y Exploración Inicial**
- Importación de datasets desde estructura local: `/Space-Titanic/Data/`
- Análisis inicial: 8,693 pasajeros en entrenamiento
- Variable objetivo: `Transported` (transportados a dimensión alternativa)

### 2. **Análisis de Valores Nulos**

#### Función de Análisis Personalizada
```python
def porcentaje_nulos(df):
    nulos = df.isnull().sum()
    porcentaje = (nulos / len(df)) * 100
    return pd.DataFrame({
        'Column': df.columns, 
        'Porcentaje Nulos': porcentaje
    }).sort_values(by='Porcentaje Nulos', ascending=False)
```

**Beneficios:**
- **Visualización clara** del estado de completitud de datos
- **Identificación prioritaria** de columnas problemáticas
- **Análisis comparativo** entre train y test

### 3. **Estrategia de Imputación Inteligente**

#### Imputación Diferenciada por Tipo
```python
for column in [train, test]:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)  # Moda para categóricas
    else:
        df[column].fillna(df[column].mean(), inplace=True)     # Media para numéricas
```

**Justificación:**
- **Variables categóricas:** Preservación de distribución natural
- **Variables numéricas:** Mantenimiento de tendencia central
- **Consistencia:** Aplicación uniforme en ambos conjuntos

### 4. **Ingeniería de Características Espaciales**

#### 4.1 Extracción de Información de Cabina
```python
# Estructura: Deck/Num/Side (ej: "B/0/P")
train[['Deck', 'NumCabin', 'Side']] = train['Cabin'].str.split('/', expand=True)
```
**Innovación:** Descomposición automática de ubicación espacial en la nave

#### 4.2 Creación de Gasto Total
```python
df['TotalSpend'] = (
    df['RoomService'].fillna(0) + 
    df['FoodCourt'].fillna(0) + 
    df['ShoppingMall'].fillna(0) + 
    df['Spa'].fillna(0) + 
    df['VRDeck'].fillna(0)
)
```
**Propósito:** Agregación de comportamiento de consumo espacial

#### 4.3 Categorización de Edad Espacial
```python
def categorize_age(age):
    if age < 18: return 'Joven'
    elif 18 <= age < 60: return 'Adulto'
    else: return 'Anciano'
```
**Contexto:** Grupos etarios relevantes para viajes espaciales

#### 4.4 Indicador de Actividad Económica
```python
df['SpentMoney'] = (df['TotalSpend'] > 0).astype(int)
```
**Hipótesis:** Pasajeros activos vs. pasivos durante el viaje

### 5. **Análisis Exploratorio Espacial**

#### Visualizaciones Clave del Contexto Sci-Fi
1. **CryoSleep vs. Transported:** Relación entre estado criogénico y transporte dimensional
2. **Deck Distribution:** Patrones de transporte por nivel de la nave
3. **Side Analysis:** Diferencias entre babor (P) y estribor (S)

#### Insights Espaciales
- **CryoSleep:** Factor crítico en probabilidad de transporte
- **Ubicación en nave:** Diferentes niveles (decks) muestran patrones únicos
- **Comportamiento de consumo:** Correlación con destino dimensional

### 6. **Preparación Avanzada de Datos**

#### Codificación con One-Hot Encoding
```python
X = pd.get_dummies(X, drop_first=True)
```
**Ventaja:** Manejo óptimo de variables categóricas sin orden jerárquico

#### Sincronización Train-Test
```python
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)
```
**Garantía:** Consistencia dimensional entre conjuntos

### 7. **Modelado con CatBoost**

#### Configuración del Modelo Principal
```python
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=0
)
```

**Justificación de CatBoost:**
- **Manejo automático de categóricas:** Ideal para datos espaciales diversos
- **Robustez a overfitting:** Regularización integrada
- **Eficiencia computacional:** Entrenamiento rápido en datasets medianos
- **Interpretabilidad:** Facilidad para análisis de importancia

### 8. **Evaluación Comprehensiva**

#### Métricas de Rendimiento
- **Accuracy Score:** Porcentaje de predicciones correctas
- **Matriz de Confusión:** Análisis detallado de errores de clasificación
- **Visualización térmica:** Heatmap para interpretación intuitiva

#### División Estratificada
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
**Beneficio:** Preservación de distribución de clases en validación

### 9. **Generación de Predicciones Dimensionales**

#### Transformación de Salida
```python
final_predictions_bool = final_predictions.astype(bool)
```
**Contexto:** Formato específico requerido por la competición (True/False)

#### Archivo de Submission
```python
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': final_predictions_bool
})
```

## 📁 Estructura del Proyecto

```
Space-Titanic/
│
├── .git/                                   # Control de versiones
├── Data/
│   ├── sample_submission.csv               # Formato de ejemplo para submissions
│   ├── test.csv                           # Dataset de prueba (4,277 pasajeros)
│   └── train.csv                          # Dataset de entrenamiento (8,693 pasajeros)
│
├── Notebook/
│   ├── catboost_info/                     # Información del modelo CatBoost
│   ├── ordenado.ipynb                     # Notebook principal con análisis completo
│   ├── submission_catboost.csv            # Predicciones finales con CatBoost
│   ├── submission_lightgbm.csv            # Predicciones con LightGBM
│   ├── submission_stacking_ridge.csv      # Predicciones con Stacking ensemble
│   └── submission.csv                     # Predicciones base
│
├── Titanic/                               # Directorio adicional
└── spaceship_titanic_readme.md           # Este archivo README
```

## 🚀 Cómo Ejecutar el Proyecto

### Prerrequisitos
```bash
pip install pandas matplotlib seaborn scikit-learn catboost lightgbm xgboost tabulate
```

### Ejecución Local
```bash
# Clonar el repositorio
git clone [repository-url]
cd Space-Titanic

# Los datos ya están en la carpeta Data/
# train.csv y test.csv incluidos

# Ejecutar el notebook
jupyter notebook Notebook/ordenado.ipynb
```

### Flujo de Ejecución
1. **Celda 1:** Carga de librerías y datasets desde `/Data/`
2. **Celda 2:** Análisis de valores nulos con función personalizada
3. **Celda 3:** Imputación inteligente por tipos de datos
4. **Celda 4:** Extracción de características de cabina
5. **Celda 5:** Ingeniería de características espaciales
6. **Celdas 6-7:** Análisis exploratorio con visualizaciones
7. **Celda 8:** Análisis de correlación
8. **Celda 9:** Modelado, evaluación y predicción final

## 📈 Resultados y Métricas

### Modelos Implementados
1. **CatBoost Classifier** (Principal)
2. **LightGBM Classifier**  
3. **Stacking with Ridge** (Ensemble)

### Modelo Principal: CatBoost Classifier
**Configuración optimizada:**
- **Iteraciones:** 1000 (balanceando rendimiento y tiempo)
- **Learning Rate:** 0.1 (convergencia estable)
- **Profundidad:** 6 (complejidad moderada)
- **Manejo automático:** Variables categóricas sin preprocesamiento

**Archivos de Salida:**
- `submission_catboost.csv` - Predicciones principales
- `submission_lightgbm.csv` - Predicciones alternativas
- `submission_stacking_ridge.csv` - Ensemble avanzado

### Insights del Contexto Espacial
1. **CryoSleep:** Factor más determinante para transporte dimensional
2. **Ubicación en nave:** Diferentes decks muestran patrones únicos
3. **Comportamiento de consumo:** Pasajeros activos vs. pasivos
4. **Características demográficas:** Patrones por grupos etarios

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque Sci-Fi
1. **Ingeniería contextual:** Características específicas para ambiente espacial
2. **Análisis dimensional:** Extracción de información de ubicación en nave
3. **Modelado especializado:** CatBoost óptimo para datos categóricos diversos
4. **Pipeline robusto:** Manejo consistente de datos no vistos

### Aspectos Únicos del Proyecto
- **Contexto narrativo:** Análisis de datos en setting de ciencia ficción
- **Características espaciales:** TotalSpend, Deck extraction, Side analysis
- **Análisis de estado:** CryoSleep como factor crítico
- **Múltiples submissions:** Comparación de diferentes enfoques

## 🎯 Posibles Mejoras

### Técnicas Avanzadas
1. **Ensemble methods:** Combinación con Random Forest, XGBoost, LightGBM
2. **Optimización de hiperparámetros:** Grid/Random Search sistemático
3. **Feature selection:** Algoritmos de selección automática
4. **Cross-validation temporal:** Si hubiera información secuencial

### Ingeniería de Características Espaciales
1. **Interacciones complejas:** Productos entre características clave
2. **Clustering de pasajeros:** Segmentación por perfiles de viaje
3. **Análisis de servicios:** Patrones detallados en RoomService, Spa, VRDeck
4. **Geolocalización espacial:** Análisis de proximidad en cabinas

## 🌌 Contexto Sci-Fi y Aplicaciones

### Relevancia del Setting Espacial
- **Viajes interestelares:** Simulación de desafíos en exploración espacial
- **Anomalías dimensionales:** Modelado de fenómenos físicos complejos
- **Gestión de pasajeros:** Optimización de recursos en naves espaciales
- **Tecnología criogénica:** Análisis de estados alterados de consciencia

### Aplicaciones en el Mundo Real
1. **Análisis de transporte:** Modelado de sistemas de transporte masivo
2. **Gestión hotelera:** Predicción de comportamiento de huéspedes
3. **Análisis de consumo:** Patrones de gasto en servicios de lujo
4. **Segmentación de clientes:** Identificación de perfiles de viajeros

## 📝 Notas de Implementación

- **Rutas locales:** Estructura adaptada para ejecución en Windows/Linux
- **Gestión de memoria:** Optimizado para datasets de tamaño medio
- **Reproducibilidad:** `random_state=42` para resultados consistentes
- **Múltiples formatos:** Diferentes submissions para comparación
- **Documentación:** Comentarios explicativos en contexto espacial

## 📞 Contacto

Para consultas técnicas, colaboraciones o discusiones sobre aplicaciones en ciencia ficción y modelado predictivo, no dudes en contactar.

---

*Este proyecto representa una exploración fascinante de machine learning aplicado a un contexto de ciencia ficción, combinando técnicas modernas de análisis de datos con narrativa espacial para crear modelos predictivos robustos y temáticamente coherentes.*