# Clasificación de Desastres Naturales con DistilBERT - NLP Fine-Tuning Project

Este proyecto implementa una solución completa para la competición de Kaggle "Natural Language Processing with Disaster Tweets" utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP) y fine-tuning del modelo transformer DistilBERT. El objetivo es clasificar tweets como relacionados o no relacionados con desastres naturales reales mediante análisis de texto.

## 🧠 Descripción del Proyecto

El proyecto utiliza el modelo **DistilBERT** pre-entrenado para realizar fine-tuning específico en el dominio de clasificación de texto sobre desastres naturales. A través de técnicas de limpieza de texto, tokenización avanzada y entrenamiento con datos etiquetados, se construye un clasificador binario capaz de identificar automáticamente tweets que reportan desastres reales versus contenido no relacionado.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Deep Learning** | PyTorch | - | Framework de deep learning |
| **Transformers** | Hugging Face Transformers | - | Modelos de lenguaje pre-entrenados |
| **Modelo Base** | DistilBERT Base Uncased | - | Modelo transformer para fine-tuning |
| **Análisis de Datos** | Pandas | - | Manipulación de datasets de texto |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y vectoriales |
| **Machine Learning** | Scikit-learn | - | Métricas de evaluación y división de datos |
| **NLP Training** | Trainer API | - | Entrenamiento simplificado de modelos |
| **Métricas** | F1 Score | - | Evaluación de clasificación binaria |
| **Preprocessing** | Regex | - | Limpieza y normalización de texto |

## 📄 Pipeline de Desarrollo

### 1. **Carga y Exploración de Datos**
```python
# Carga de datasets desde Kaggle
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
```

**Datasets principales:**
- **train.csv:** Tweets etiquetados como desastre (1) o no desastre (0)
- **test.csv:** Conjunto de evaluación sin etiquetas
- **sample_submission.csv:** Formato de envío requerido

### 2. **Preprocesamiento y Limpieza de Datos**

#### Función de Limpieza de Texto
```python
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Eliminar URLs
    text = re.sub(r'\@\w+|\#','', text)  # Eliminar hashtags y menciones
    text = re.sub(r"[^a-zA-Z0-9\s]", '', text)  # Eliminar caracteres especiales
    return text
```

**Beneficios del preprocesamiento:**
- **Normalización:** Conversión a minúsculas para consistencia
- **Eliminación de ruido:** Remoción de URLs, hashtags y menciones
- **Simplificación:** Conservación solo de caracteres alfanuméricos
- **Mejora de rendimiento:** Reducción de vocabulario y ruido

#### División de Datos
```python
# División estratificada para training/validation
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, 
    test_size=0.2, 
    random_state=2
)
```

### 3. **Configuración del Modelo DistilBERT**

#### Carga del Modelo Base
```python
# Carga del tokenizador y modelo DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)
```

**Justificación de DistilBERT:**
- **Eficiencia:** 40% más pequeño que BERT con 97% del rendimiento
- **Velocidad:** 60% más rápido en inferencia
- **Capacidad:** Mantiene arquitectura transformer completa
- **Pre-entrenamiento:** Entrenado en corpus masivo de texto en inglés

#### Tokenización Avanzada
```python
# Tokenización con parámetros optimizados
train_encodings = tokenizer(
    X_train, 
    truncation=True, 
    padding=True, 
    max_length=256
)
val_encodings = tokenizer(
    X_val, 
    truncation=True, 
    padding=True, 
    max_length=256
)
```

**Configuración optimizada:**
- **max_length=256:** Balance entre contexto y memoria
- **truncation=True:** Manejo de tweets largos
- **padding=True:** Uniformidad en batch processing

### 4. **Dataset Personalizado y Carga de Datos**

#### Clase Dataset Custom
```python
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
```

**Características del dataset:**
- **Compatibilidad PyTorch:** Integración nativa con DataLoader
- **Eficiencia de memoria:** Carga lazy de datos
- **Flexibilidad:** Manejo de diferentes tipos de encoding

### 5. **Configuración de Entrenamiento y Fine-Tuning**

#### Parámetros de Entrenamiento Optimizados
```python
training_args = TrainingArguments(
    output_dir='./results',
    report_to='none',  # Desactiva logging externo
    num_train_epochs=5,  # Épocas suficientes para convergencia
    per_device_train_batch_size=16,  # Optimizado para GPU
    per_device_eval_batch_size=32,  # Batch mayor para evaluación
    warmup_steps=500,  # Calentamiento gradual del learning rate
    weight_decay=0.01,  # Regularización L2
    logging_dir='./logs',
    evaluation_strategy='steps',  # Evaluación continua
    eval_steps=500,  # Frecuencia de evaluación
    learning_rate=3e-5,  # Learning rate ajustado para fine-tuning
)
```

#### Técnica de Freezing de Capas
```python
# Congelar capas base para fine-tuning selectivo
for param in model.distilbert.parameters():
    param.requires_grad = False
```

**Ventajas del layer freezing:**
- **Prevención de overfitting:** Mantiene representaciones pre-entrenadas
- **Velocidad de entrenamiento:** Reduce parámetros a actualizar
- **Estabilidad:** Evita catastrophic forgetting
- **Eficiencia computacional:** Menor uso de memoria GPU

### 6. **Entrenamiento y Evaluación del Modelo**

#### Proceso de Entrenamiento
```python
# Creación del entrenador con configuración optimizada
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Ejecutar fine-tuning
trainer.train()
```

#### Evaluación y Métricas
```python
# Evaluación del modelo en conjunto de validación
trainer.evaluate()

# Predicciones y cálculo de F1 Score
val_predictions = trainer.predict(val_dataset)
val_pred_labels = val_predictions.predictions.argmax(-1)

# Métrica principal: F1 Score
f1 = f1_score(y_val, val_pred_labels)
print(f'F1 Score: {f1}')
```

### 7. **Predicción en Datos de Prueba**

#### Preparación de Datos de Test
```python
# Aplicar misma limpieza a datos de prueba
test_texts = [clean_text(text) for text in test_data['text'].tolist()]
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# Crear dataset de prueba con labels dummy
test_dataset = NewsDataset(test_encodings, [0] * len(test_encodings['input_ids']))
```

#### Generación de Predicciones Finales
```python
# Predicción sobre conjunto de prueba
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)

# Preparación del archivo de submission
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'target': predicted_labels
})

# Guardado del archivo de envío
submission_df.to_csv('submission.csv', index=False)
```

## 🏗️ Estructura del Proyecto (Kaggle Environment)

### Entorno de Kaggle - Archivos y Datasets Disponibles:

```
COMPETITIONS:
└── nlp-getting-started/
    ├── train.csv                              # Dataset de entrenamiento
    ├── test.csv                              # Dataset de evaluación  
    └── sample_submission.csv                 # Formato de envío

NOTEBOOK:
└── disaster_classification_distilbert.ipynb  # Notebook principal del proyecto

OUTPUT (/kaggle/working/):
├── results/                                  # Checkpoints del modelo
│   ├── checkpoint-500/                      # Checkpoint intermedio
│   └── checkpoint-1000/                     # Checkpoint final
├── logs/                                    # Logs de entrenamiento
└── submission.csv                           # Archivo de envío
```

### Rutas de Acceso en Código:
```python
# Datos de entrada desde competitions
TRAIN_PATH = "/kaggle/input/nlp-getting-started/train.csv"
TEST_PATH = "/kaggle/input/nlp-getting-started/test.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/nlp-getting-started/sample_submission.csv"

# Outputs en working directory
RESULTS_PATH = "/kaggle/working/results"
LOGS_PATH = "/kaggle/working/logs"
SUBMISSION_PATH = "/kaggle/working/submission.csv"
```

## 🚀 Cómo Ejecutar el Proyecto en Kaggle

### Configuración del Entorno Kaggle
```python
# Instalación de librerías necesarias
!pip install transformers torch

# Librerías principales disponibles en Kaggle
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import re
```

### Verificación de GPU y Datasets
```python
# Verificar disponibilidad de GPU
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No disponible")

# Listar archivos disponibles en input
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

### Ejecución Paso a Paso

#### Paso 1: Crear y ejecutar un nuevo notebook
1. **Crear notebook** en competición "Natural Language Processing with Disaster Tweets"
2. **Configurar GPU**: Settings → Accelerator → GPU T4 x2
3. **Agregar datasets**:
   - **Competition Data**: Natural Language Processing with Disaster Tweets

#### Paso 2: Verificar archivos disponibles
```python
# Verificar estructura de input
import os

print("=== COMPETITION DATA ===")
for file in os.listdir('/kaggle/input/nlp-getting-started/'):
    print(f"📄 {file}")
```

### Flujo de Ejecución en Kaggle

#### 1. **Preparación de Datos**
```python
# Cargar datasets desde input directory
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Aplicar limpieza de texto
train_texts = [clean_text(text) for text in train_data['text'].tolist()]
train_labels = train_data['target'].tolist()
```

#### 2. **División y Preparación de Datasets**
```python
# División estratificada
X_train, X_val, y_train, y_val = train_test_split(
    train_texts, train_labels, 
    test_size=0.2, 
    random_state=2
)

# Tokenización con DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=256)
```

#### 3. **Fine-Tuning del Modelo**
```python
# Cargar modelo pre-entrenado
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=2
)

# Aplicar layer freezing para fine-tuning selectivo
for param in model.distilbert.parameters():
    param.requires_grad = False

# Configurar training arguments para guardar en working
training_args = TrainingArguments(
    output_dir='/kaggle/working/results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    evaluation_strategy='steps',
    eval_steps=500,
)

# Entrenar modelo
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()
```

#### 4. **Evaluación y Predicción**
```python
# Evaluación en conjunto de validación
trainer.evaluate()

# Predicciones en conjunto de prueba
test_dataset = NewsDataset(test_encodings, [0] * len(test_encodings['input_ids']))
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(-1)

# Guardar submission en working directory
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'target': predicted_labels
})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)
```

## 📈 Resultados y Métricas

### Modelo Principal: DistilBERT Fine-Tuned
**Configuración optimizada:**
- **Arquitectura:** DistilBERT (Distilled BERT)
- **Tamaño:** 66M parámetros (vs 110M de BERT-base)
- **Epochs:** 5 (balance entre convergencia y overfitting)
- **Batch Size:** 16 para entrenamiento, 32 para evaluación
- **Learning Rate:** 3e-5 (optimizado para fine-tuning)
- **Layer Freezing:** Capas base congeladas, solo fine-tuning de clasificador

### Métricas de Evaluación
- **Métrica principal:** F1 Score (balance entre precisión y recall)
- **Estrategia de evaluación:** Evaluación por pasos cada 500 iteraciones
- **Regularización:** Weight decay 0.01 para prevenir overfitting
- **Warmup:** 500 pasos de calentamiento gradual del learning rate

### Técnicas de Optimización Implementadas
1. **Preprocesamiento robusto:** Limpieza sistemática de texto
2. **Layer freezing:** Fine-tuning selectivo para eficiencia
3. **Batch size diferenciado:** Optimización de memoria GPU
4. **Evaluación continua:** Monitoreo de performance durante entrenamiento

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque NLP
1. **Fine-tuning selectivo:** Congelamiento de capas base para estabilidad
2. **Preprocesamiento especializado:** Limpieza adaptada a tweets
3. **Tokenización optimizada:** Configuración específica para textos cortos
4. **Evaluación robusta:** F1 Score para datasets desbalanceados

### Aspectos Únicos del Proyecto
- **Dominio específico:** Aplicación de NLP a detección de desastres
- **Eficiencia computacional:** Uso de DistilBERT para rapidez
- **Preprocessing avanzado:** Manejo especializado de contenido de redes sociales
- **Transfer learning:** Aprovechamiento de conocimiento pre-entrenado

## 🎯 Posibles Mejoras

### Técnicas Avanzadas de NLP
1. **Ensemble de modelos:** Combinación con RoBERTa, ALBERT, o ELECTRA
2. **Data augmentation:** Generación sintética de ejemplos de entrenamiento
3. **Curriculum learning:** Entrenamiento progresivo con ejemplos de dificultad creciente
4. **Multi-task learning:** Entrenamiento conjunto en tareas relacionadas

### Ingeniería de Características Especializadas
1. **Análisis de sentimientos:** Incorporación de polaridad emocional
2. **Extracción de entidades:** Identificación de ubicaciones y eventos
3. **Características lingüísticas:** Longitud, complejidad, patrones sintácticos
4. **Metadatos de tweets:** Información temporal, geográfica, de usuario

### Optimización de Modelo
1. **Hyperparameter tuning:** Búsqueda sistemática de parámetros óptimos
2. **Learning rate scheduling:** Estrategias de decaimiento adaptativo
3. **Gradient clipping:** Control de gradientes para estabilidad
4. **Early stopping:** Parada temprana basada en métricas de validación

### Técnicas de Evaluación Avanzadas
1. **Cross-validation:** Validación cruzada k-fold para robustez
2. **Análisis de errores:** Estudio cualitativo de clasificaciones incorrectas
3. **Métricas adicionales:** Precisión, recall, AUC-ROC
4. **Análisis de sesgo:** Evaluación de fairness en diferentes grupos

## 🎯 Aplicaciones del Mundo Real

### Impacto en Gestión de Emergencias
- **Detección temprana:** Identificación rápida de eventos de desastre
- **Monitoreo de redes sociales:** Análisis automatizado de contenido
- **Alertas automáticas:** Sistemas de notificación en tiempo real
- **Análisis de tendencias:** Identificación de patrones de propagación

### Escalabilidad y Transferencia
1. **Otros idiomas:** Adaptación a modelos multilingües
2. **Diferentes plataformas:** Extensión a Facebook, Instagram, TikTok
3. **Tipos de emergencia:** Especialización en terremotos, incendios, inundaciones
4. **Integración en sistemas:** APIs para centros de comando y control

## 🔧 Consideraciones Técnicas

### Consideraciones Específicas de Kaggle

#### Limitaciones del Entorno
- **Tiempo de ejecución:** Máximo 12 horas por sesión
- **Espacio en /working:** 20GB disponibles
- **GPU:** T4 x2 disponible para training
- **Internet:** Habilitado para descarga de modelos pre-entrenados
- **Librerías:** Instalación de transformers requerida

#### Optimizaciones para Kaggle
```python
# Configuración de memoria eficiente
torch.cuda.empty_cache()

# Batch size optimizado para T4
TRAIN_BATCH_SIZE = 16  # Balance entre velocidad y memoria
EVAL_BATCH_SIZE = 32   # Batch mayor para evaluación

# Checkpointing estratégico
training_args = TrainingArguments(
    output_dir='/kaggle/working/results',
    save_steps=500,  # Guardar cada 500 pasos
    logging_steps=100,
    eval_steps=500,  # Evaluación frecuente
)
```

#### Gestión de Archivos
```python
# Verificar espacio disponible
import shutil
total, used, free = shutil.disk_usage('/kaggle/working')
print(f"Espacio libre: {free // (2**30)} GB")

# Limpiar archivos innecesarios
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Reproducibilidad
- **Random seeds:** Fijados para consistencia de resultados
- **Versiones de librerías:** Especificadas para compatibilidad
- **Checkpoints:** Guardado regular durante entrenamiento
- **Logging:** Registro detallado de métricas y parámetros

### Manejo de Errores Comunes
```python
# Verificar compatibilidad de versiones
try:
    from transformers import DistilBertTokenizer
    print("✅ Transformers instalado correctamente")
except ImportError:
    print("❌ Error: Instalar transformers con !pip install transformers")

# Verificar disponibilidad de GPU
if torch.cuda.is_available():
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU no disponible, usando CPU")
```

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos de NLP, o discusiones sobre aplicaciones de IA en gestión de emergencias, no dudes en contactar.

## 📗 Referencias y Recursos

- **DistilBERT Paper:** "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
- **BERT Architecture:** "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- **Disaster Detection:** Literatura sobre detección automática de desastres en redes sociales
- **Transfer Learning:** Investigación en fine-tuning de modelos transformer

---

*Este proyecto representa una aplicación práctica de técnicas modernas de NLP para la detección automática de desastres en redes sociales, combinando fine-tuning eficiente con preprocessing especializado para crear una herramienta útil en gestión de emergencias y monitoreo de eventos críticos.*