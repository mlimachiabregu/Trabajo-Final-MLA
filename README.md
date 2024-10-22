🚀 Evaluacion de Clasificación con PyCaret, MLflow y FastAPI

Este Evaluacion se centra en la creación, evaluación y despliegue de un modelo de clasificación utilizando PyCaret, MLflow y FastAPI. Los datos provienen del desafío de Kaggle: Playground Series - Season 4, Episode 1. El objetivo es entrenar un modelo para predecir resultados, gestionando versiones y experimentos con MLflow, y desplegar un servicio de predicción con FastAPI y subir las capturas de la ejecucion de la solucion se pide completar el notebook de mlflow y el app.

🛠️ Componentes Clave del Proyecto

📁 Carga de Archivos de Entrenamiento y Prueba

Los archivos train.csv y test.csv se utilizan para entrenar y evaluar el modelo.
Los datos se cargan a través de la API con FastAPI en formato CSV y se procesan en pandas para generar DataFrames.

🧪 Entrenamiento del Modelo con PyCaret

PyCaret facilita la experimentación con varios modelos de clasificación.
Preprocesamiento automático: Manejo de datos faltantes, normalización, codificación categórica, etc.
Comparación de modelos: Compara modelos como Random Forest, SVM, XGBoost.
Tuning de Hiperparámetros: Ajuste automático de hiperparámetros.
Integración con MLflow: Registro automático de experimentos, métricas y modelos.

📝 Gestión del Ciclo de Vida del Modelo con MLflow

MLflow gestiona los experimentos y modelos entrenados:
Registro automático de modelos y parámetros probados.
Almacenamiento de métricas de rendimiento como Precisión, Recall, F1-Score, AUC, entre otras.
Versionado de modelos: Identificación del mejor modelo con facilidad.
Comparación de modelos en función de las métricas obtenidas.
Gestión clara de versiones y artefactos.

🌐 Despliegue del Modelo con FastAPI

Después de entrenar y registrar el modelo con MLflow, se despliega mediante FastAPI.
Endpoints principales:

    /upload: Carga de conjuntos de datos en formato CSV.
    /train: Entrenamiento del modelo con PyCaret.
    /predict: Predicción de nuevos datos utilizando el mejor modelo almacenado.

📋 Cómo Ejecutar el Proyecto
Instalación de Dependencias:

pip install -r requirements.txt
Ejecutar FastAPI:

uvicorn main:app --reload
Endpoints Disponibles:

  /upload: Subir datasets para el entrenamiento.
  /train: Entrenar el modelo con PyCaret.
  /predict: Hacer predicciones con el modelo entrenado.

🖥️ Tecnologías Utilizadas

PyCaret: Para el entrenamiento y comparación de modelos de clasificación.
MLflow: Para gestionar el ciclo de vida de los modelos.
FastAPI: Para el despliegue de un servicio de predicción basado en el modelo.

¡Gracias por revisar el proyecto! 🌟
