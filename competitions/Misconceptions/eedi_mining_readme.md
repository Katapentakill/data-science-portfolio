# EEDI Mining Misconceptions in Mathematics - NLP Fine-Tuning Project

Este proyecto implementa una solución completa para la competición de Kaggle "EEDI - Mining Misconceptions in Mathematics" utilizando técnicas avanzadas de procesamiento de lenguaje natural (NLP) y fine-tuning de modelos transformer. El objetivo es identificar y mapear conceptos erróneos matemáticos en respuestas incorrectas de estudiantes mediante el análisis de preguntas y opciones de respuesta.

## 🧠 Descripción del Proyecto

El proyecto utiliza el modelo **Flan-T5** pre-entrenado para realizar fine-tuning específico en el dominio de educación matemática. A través de la generación de prompts estructurados y el entrenamiento con datos etiquetados, se construye un sistema inteligente capaz de identificar automáticamente los conceptos erróneos subyacentes en las respuestas incorrectas de estudiantes.

## 📊 Tecnologías Utilizadas

| Categoría | Tecnología | Versión | Propósito |
|-----------|------------|---------|-----------|
| **Lenguaje** | Python | 3.x | Lenguaje principal de desarrollo |
| **Deep Learning** | PyTorch | - | Framework de deep learning |
| **Transformers** | Hugging Face Transformers | - | Modelos de lenguaje pre-entrenados |
| **Modelo Base** | Flan-T5 Base | - | Modelo transformer para fine-tuning |
| **Análisis de Datos** | Pandas | - | Manipulación de datasets educativos |
| **Análisis de Datos** | NumPy | - | Operaciones numéricas y vectoriales |
| **Machine Learning** | Scikit-learn | - | Métricas de evaluación y preprocesamiento |
| **Datasets** | Hugging Face Datasets | - | Manejo eficiente de datos para training |
| **NLP Training** | Trainer API | - | Entrenamiento simplificado de modelos |
| **Métricas** | Cosine Similarity | - | Cálculo de similitud semántica |
| **Utilidades** | Tabulate | - | Formateo de resultados |
| **Serialización** | JSON | - | Almacenamiento de prompts estructurados |

## 🔄 Pipeline de Desarrollo

### 1. **Carga y Exploración de Datos**
```python
# Carga de datasets desde Kaggle
test = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv")
train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
misconceptions_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")
```

**Datasets principales:**
- **train.csv:** Preguntas matemáticas con conceptos erróneos etiquetados
- **test.csv:** Conjunto de evaluación sin etiquetas
- **misconceptions_mapping.csv:** Mapeo de IDs a descripciones de conceptos erróneos
- **sample_submission.csv:** Formato de envío requerido

### 2. **Preprocesamiento y Limpieza de Datos**

#### Conversión de Tipos de Datos
```python
def convert_to_int(column):
    return column.apply(lambda x: int(x) if pd.notna(x) else pd.NA)

# Conversión a Int64 para manejo correcto de valores nulos
for column in ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']:
    train[column] = train[column].astype('Int64')
```

**Beneficios:**
- **Manejo robusto de NaN:** Preservación de valores faltantes
- **Tipado consistente:** Evita errores en operaciones posteriores
- **Compatibilidad:** Preparación para análisis estadístico

#### Filtrado y Análisis de Frecuencias
```python
# Filtrar filas con al menos un concepto erróneo
filtered_train = train[
    train['MisconceptionAId'].notna() |
    train['MisconceptionBId'].notna() |
    train['MisconceptionCId'].notna() |
    train['MisconceptionDId'].notna()
]
```

### 3. **Generación de Prompts Estructurados**

#### Formato de Prompt Educativo
```python
prompt = (
    f"Question: {row['QuestionText']}\n"
    "Possible Answers:\n"
    f"A: {row['AnswerAText']}\n"
    f"B: {row['AnswerBText']}\n"
    f"C: {row['AnswerCText']}\n"
    f"D: {row['AnswerDText']}\n"
    f"Correct Answer: {row['CorrectAnswer']}\n"
)
```

**Características del diseño:**
- **Estructura clara:** Formato pregunta-opciones-respuesta correcta
- **Contexto educativo:** Adaptado para análisis de conceptos erróneos
- **Consistencia:** Formato uniforme para todo el dataset

#### Construcción de Completions
```python
misconceptions = []
for option in ['A', 'B', 'C', 'D']:
    misconception_id = row[f'Misconception{option}Id']
    if pd.notna(misconception_id) and misconception_id in misconceptions_dict:
        misconceptions.append(f"- Option {option}: {{id={int(misconception_id)}}} {misconceptions_dict[misconception_id]}")

completion = "\n".join(misconceptions) if misconceptions else "No misconceptions available."
```

**Innovación:**
- **Mapeo automático:** ID a descripción textual de conceptos erróneos
- **Estructura jerárquica:** Organización por opción de respuesta
- **Manejo de casos edge:** Gestión de misconceptions faltantes

### 4. **Fine-Tuning del Modelo Flan-T5**

#### Configuración del Modelo Base
```python
model_path = "/kaggle/input/flan-t5-base-v4"
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)
```

**Justificación de Flan-T5:**
- **Instrucción-following:** Diseñado para seguir instrucciones complejas
- **Versatilidad:** Excelente para tareas de generación condicionada
- **Tamaño eficiente:** Balance entre capacidad y recursos computacionales
- **Fine-tuning friendly:** Arquitectura optimizada para adaptación específica

#### Preparación de Datasets
```python
# División estratificada para training/validation
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

# Conversión a formato Hugging Face
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
```

### 5. **Entrenamiento y Evaluación**

#### Configuración de Training
```python
training_args = TrainingArguments(
    output_dir='/kaggle/working/flan-t5-finetuned',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch'
)
```

#### Generación de Predicciones
```python
model.eval()
for item in selected_prompts:
    inputs = tokenizer(item["prompt"], return_tensors='pt', padding=True, truncation=True, max_length=256).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs)
        pred_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### 6. **Métrica de Evaluación: mAP@25**

#### Cálculo de Embeddings Semánticos
```python
def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        outputs = model.base_model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1)
```

#### Implementación de mAP@25
```python
def compute_map_at_25(predictions, labels):
    pred_vectors = get_embeddings(predictions).cpu().numpy()
    label_vectors = get_embeddings(labels).cpu().numpy()
    
    cosine_similarities = cosine_similarity(pred_vectors, label_vectors)
    
    map_at_25 = 0
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        top_25_preds = sorted(range(len(cosine_similarities[i])), 
                             key=lambda k: cosine_similarities[i][k], reverse=True)[:25]
        # Cálculo de precisión promedio
        # ... [lógica de relevancia y ranking]
```

**Características de la métrica:**
- **Ranking-based:** Evalúa la calidad del ordenamiento de predicciones
- **Top-K evaluation:** Foco en las 25 mejores predicciones
- **Similitud semántica:** Uso de cosine similarity para comparación
- **Educación-específica:** Adaptada para conceptos erróneos matemáticos

## 📁 Estructura del Proyecto (Kaggle Environment)

### Entorno de Kaggle - Archivos y Datasets Disponibles:

```
COMPETITIONS:
└── eedi-mining-misconceptions-in-mathematics/
    ├── train.csv                              # Dataset de entrenamiento
    ├── test.csv                              # Dataset de evaluación  
    ├── sample_submission.csv                 # Formato de envío
    └── misconception_mapping.csv             # Mapeo ID-concepto erróneo

MODELS:
└── flan-t5-base-v4/
    ├── .gitattributes                        # Atributos de Git
    ├── README.md                             # Documentación del modelo
    ├── config.json                           # Configuración del modelo
    ├── flax_model.msgpack                    # Modelo en formato Flax
    ├── generation_config.json                # Configuración de generación
    ├── model.safetensors                     # Modelo principal
    ├── pytorch_model.bin                     # Modelo PyTorch
    ├── special_tokens_map.json               # Mapeo de tokens especiales
    ├── spiece.model                          # Modelo SentencePiece
    ├── tf_model.h5                          # Modelo TensorFlow
    ├── tokenizer.json                        # Tokenizador
    └── tokenizer_config.json                 # Configuración del tokenizador

NOTEBOOK:
└── misconception1_flan.ipynb                 # Notebook principal del proyecto

OUTPUT (/kaggle/working/):
├── fine_tuning_prompts.json                  # Prompts generados
├── flan-t5-finetuned/                       # Modelo fine-tuneado
└── submission.csv                            # Archivo de envío
```

### Rutas de Acceso en Código:
```python
# Datos de entrada desde competitions
TRAIN_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv"
TEST_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/test.csv"
MISCONCEPTIONS_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/sample_submission.csv"

# Modelo pre-entrenado desde models
MODEL_PATH = "/kaggle/input/flan-t5-base-v4"
TOKENIZER_PATH = "/kaggle/input/flan-t5-base-v4"

# Outputs en working directory
FINETUNED_MODEL_PATH = "/kaggle/working/flan-t5-finetuned"
PROMPTS_PATH = "/kaggle/working/fine_tuning_prompts.json"
SUBMISSION_PATH = "/kaggle/working/submission.csv"
```

## 🚀 Cómo Ejecutar el Proyecto en Kaggle

### Configuración del Entorno Kaggle
```python
# Las librerías principales ya están disponibles en Kaggle
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
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
1. **Fork del notebook** `misconception1_flan.ipynb`
2. **Configurar GPU**: Settings → Accelerator → GPU T4 x2
3. **Agregar datasets**:
   - **Competition Data**: EEDI - Mining Misconceptions in Mathematics
   - **Model**: flan-t5-base-v4 (incluye todos los archivos del modelo)

#### Paso 2: Verificar archivos disponibles
```python
# Verificar estructura de input
import os

print("=== COMPETITION DATA ===")
for file in os.listdir('/kaggle/input/eedi-mining-misconceptions-in-mathematics/'):
    print(f"📄 {file}")

print("\n=== MODEL FILES ===") 
for file in os.listdir('/kaggle/input/flan-t5-base-v4/'):
    print(f"🤖 {file}")
```

### Flujo de Ejecución en Kaggle

#### 1. **Preparación de Datos**
```python
# Cargar datasets desde input directory
train = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv")
misconceptions_mapping = pd.read_csv("/kaggle/input/eedi-mining-misconceptions-in-mathematics/misconception_mapping.csv")

# Convertir tipos y limpiar datos
for column in ['MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']:
    train[column] = train[column].astype('Int64')
```

#### 2. **Generación de Prompts**
```python
# Crear diccionario de mapeo
misconceptions_dict = {row['MisconceptionId']: row['MisconceptionName'] 
                      for index, row in misconceptions_mapping.iterrows()}

# Generar y guardar prompts en working directory
prompts = []
for index, row in train.iterrows():
    # ... [lógica de generación de prompts]

# Guardar en /kaggle/working/
with open('/kaggle/working/fine_tuning_prompts.json', 'w') as f:
    json.dump(prompts, f, indent=4)
```

#### 3. **Fine-Tuning del Modelo**
```python
# Cargar modelo base desde input/models
model_path = "/kaggle/input/flan-t5-base-v4"
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Preparar datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Configurar training arguments para guardar en working
training_args = TrainingArguments(
    output_dir='/kaggle/working/flan-t5-finetuned',
    # ... otros parámetros
)

# Entrenar modelo
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
trainer.train()
```

#### 4. **Evaluación y Predicción**
```python
# Cargar modelo fine-tuneado desde working directory
model = T5ForConditionalGeneration.from_pretrained("/kaggle/working/flan-t5-finetuned")
model.to("cuda")  # Usar GPU de Kaggle

# Generar predicciones
predictions = []
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
    outputs = model.generate(**inputs)
    predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Guardar submission en working directory
submission.to_csv('/kaggle/working/submission.csv', index=False)
```

## 📈 Resultados y Métricas

### Modelo Principal: Flan-T5 Fine-Tuned
**Configuración optimizada:**
- **Arquitectura:** T5 (Text-to-Text Transfer Transformer)
- **Tamaño:** Base (220M parámetros)
- **Epochs:** 3 (balance entre overfitting y convergencia)
- **Batch Size:** 8 (optimizado para GPU disponible)
- **Learning Rate:** Adaptativo según Trainer defaults

### Métricas de Evaluación
- **Métrica principal:** mAP@25 (Mean Average Precision at 25)
- **Similitud semántica:** Cosine similarity entre embeddings
- **Precisión:** Evaluación en top-25 predicciones
- **Análisis cualitativo:** Revisión manual de conceptos erróneos identificados

### Insights Educativos
1. **Patrones de conceptos erróneos:** Identificación de misconceptions comunes
2. **Distribución por temas:** Análisis de áreas matemáticas problemáticas
3. **Correlaciones:** Relaciones entre tipos de preguntas y errores específicos
4. **Calidad de predicciones:** Evaluación de coherencia semántica

## 🔬 Innovaciones Técnicas

### Fortalezas del Enfoque NLP
1. **Fine-tuning especializado:** Adaptación específica al dominio educativo
2. **Prompts estructurados:** Diseño optimizado para comprensión de contexto
3. **Embeddings semánticos:** Representación vectorial de conceptos erróneos
4. **Evaluación ranking-based:** Métrica adaptada al problema específico

### Aspectos Únicos del Proyecto
- **Dominio educativo:** Aplicación de NLP a educación matemática
- **Identificación automática:** Detección de misconceptions sin supervisión directa
- **Transferencia de conocimiento:** Leverage de Flan-T5 pre-entrenado
- **Métrica especializada:** mAP@25 para evaluación de relevancia

## 🎯 Posibles Mejoras

### Técnicas Avanzadas de NLP
1. **Ensemble de modelos:** Combinación con BERT, RoBERTa, o GPT-variants
2. **Prompt engineering:** Optimización sistemática de formatos de prompt
3. **Data augmentation:** Generación sintética de ejemplos de entrenamiento
4. **Knowledge distillation:** Transferencia desde modelos más grandes

### Ingeniería de Características Educativas
1. **Análisis de dificultad:** Incorporación de métricas de complejidad de preguntas
2. **Taxonomía de errores:** Clasificación jerárquica de conceptos erróneos
3. **Patrones temporales:** Análisis de evolución de misconceptions
4. **Contexto curricular:** Integración de información de currículo matemático

### Optimización de Modelo
1. **Hyperparameter tuning:** Grid search sistemático
2. **Learning rate scheduling:** Estrategias de decaimiento adaptativo
3. **Regularization techniques:** Dropout, weight decay optimizados
4. **Multi-task learning:** Entrenamiento conjunto en tareas relacionadas

## 🎓 Aplicaciones Educativas

### Impacto en Educación Matemática
- **Diagnóstico automático:** Identificación rápida de conceptos erróneos
- **Feedback personalizado:** Respuestas específicas a errores comunes
- **Análisis curricular:** Detección de áreas problemáticas en programas
- **Formación docente:** Herramienta para entender patrones de error estudiantil

### Escalabilidad y Transferencia
1. **Otras disciplinas:** Adaptación a ciencias, física, química
2. **Diferentes idiomas:** Extensión a entornos multilingües
3. **Niveles educativos:** Aplicación desde primaria hasta universitario
4. **Sistemas adaptativos:** Integración en plataformas de aprendizaje personalizado

## 🔧 Consideraciones Técnicas

### Consideraciones Específicas de Kaggle

#### Limitaciones del Entorno
- **Tiempo de ejecución:** Máximo 12 horas por sesión
- **Espacio en /working:** 20GB disponibles
- **GPU:** T4 x2 disponible para training
- **Internet:** Deshabilitado durante competición
- **Modelos pre-cargados:** Usar datasets de input para modelos

#### Optimizaciones para Kaggle
```python
# Configuración de memoria eficiente
torch.cuda.empty_cache()

# Batch size optimizado para T4
BATCH_SIZE = 8  # Ajustar según memoria disponible

# Checkpointing frecuente
training_args = TrainingArguments(
    output_dir='/kaggle/working/checkpoints',
    save_steps=500,  # Guardar cada 500 pasos
    logging_steps=100,
    dataloader_num_workers=2  # Optimizado para Kaggle
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

## 📞 Contacto y Colaboración

Para consultas técnicas, colaboraciones en proyectos educativos, o discusiones sobre aplicaciones de NLP en educación matemática, no dudes en contactar.

## 🔗 Referencias y Recursos

- **Flan-T5 Paper:** "Scaling Instruction-Finetuned Language Models"
- **T5 Architecture:** "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **Educational NLP:** Literatura sobre aplicaciones de IA en educación
- **Misconceptions Research:** Investigación en conceptos erróneos matemáticos

---

*Este proyecto representa una aplicación innovadora de técnicas de NLP modernas al campo de la educación matemática, combinando fine-tuning especializado con análisis semántico para crear una herramienta poderosa de identificación automática de conceptos erróneos estudiantiles.*