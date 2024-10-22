import pandas as pd  # Importamos pandas para trabajar con DataFrames.
import mlflow  # Importamos mlflow para interactuar con los modelos registrados en el servidor de MLflow.
from fastapi import FastAPI  # Importamos FastAPI para crear nuestra API.
import uvicorn  # Importamos uvicorn, que es el servidor que ejecutará la aplicación FastAPI.
from pydantic import BaseModel  # Importamos BaseModel de Pydantic para definir la estructura de los datos de entrada.

# Inicializamos la aplicación FastAPI.
app = FastAPI()

# Configuramos la conexión con el servidor de MLflow para cargar el modelo.
# mlflow.set_tracking_uri indica la ubicación donde se almacenan los experimentos y modelos.
# En este caso, estamos utilizando una instancia de MLflow local que corre en http://127.0.0.1:5000.
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Cargamos el modelo de producción de MLflow que se ha registrado con el nombre 'rent_model'.
# El modelo ya está cargado y listo para ser utilizado de forma constante.
model = mlflow.pyfunc.load_model("models:/Credito_Prod/Production")

# Definimos un modelo de datos usando Pydantic, que especifica la estructura de los datos de entrada.
# Los datos que el usuario envíe a la API deben seguir esta estructura (tipo de dato y valor por defecto).
class Data(BaseModel):
    CustomerId: int=1
    CreditScore: int = 400
    Geography:  str = 'Germany'
    Gender: str = 'Masculino'
    Age: int = 18
    Tenure: int =2
    Balance: float
    NumOfProducts: int=0
    HasCrCard: int = 0
    IsActiveMember: int=1   
    EstimatedSalary: float =100.00
 #   Exited: int=1 


# Función para convertir los datos JSON recibidos en un DataFrame.
# Esto es necesario porque el modelo espera recibir un DataFrame como entrada para hacer predicciones.
def create_dataframe(data: Data):
    # Convertimos los datos recibidos a un diccionario y luego lo pasamos a un DataFrame.
    # Al indexarlo con [0], creamos un DataFrame con una sola fila.
    df = pd.DataFrame(dict(data), index=[0])

    # Renombramos las columnas del DataFrame para que coincidan con los nombres que espera el modelo.
    # Esto es importante para que el modelo pueda mapear correctamente los datos de entrada.
    df = df.rename(columns={
                            'CustomerId': 'CustomerId',
                            'CreditScore': 'CreditScore',
                            'Geography': 'Geography',
                            'Gender': 'Gender',
                            'Age': 'Age',
                            'Balance': 'Balance',
                            'NumOfProducts': 'NumOfProducts',
                            'HasCrCard': 'HasCrCard',
                            'EstimatedSalary': 'EstimatedSalary'}
                         #   'Exited': 'Exited'}
                            )
    return df

# Definimos el endpoint POST '/predict' que manejará las peticiones de predicción.
# El decorador @app.post indica que esta función se ejecutará cuando se realice una petición POST a /predict.
@app.post("/predict")
def predict(data: Data):
    # Llamamos a la función create_dataframe para convertir los datos de entrada en un DataFrame.
    data = create_dataframe(data)

    # Usamos el modelo cargado desde MLflow para hacer una predicción basada en los datos recibidos.
    prediction = model.predict(data)

    # Retornamos la predicción como un diccionario JSON, donde la clave es 'prediction'
    # y el valor es la primera (y única) predicción generada por el modelo.
    return {'prediction': prediction[0].item()}

# Punto de entrada de la aplicación.
# Si este archivo se ejecuta directamente, se iniciará el servidor FastAPI en el host 127.0.0.1 (localhost) y en el puerto 8000.
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
