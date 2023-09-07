#Importación de bibliotecas: 
#Se importa las bibliotecas necesarias, incluyendo Pandas, NumPy y varias funciones y clases relacionadas con la 
#selección de modelos y métricas de regresión de scikit-learn.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Lectura de datos
#Se lee un conjunto de datos desde un archivo CSV llamado "Student_Performance.csv". Se convierte los valores categóricos 
#en la columna "Extracurricular Activities" de "Yes" a 1 y de "No" a 0. Esto es necesario para que el algoritmo KNN pueda 
#manejar estos valores categóricos como numéricos. Se divide el conjunto de datos en dos partes -> X: Contiene todas las características 
#excepto la columna "Performance Index". y: Contiene la columna "Performance Index", que es la variable objetivo que se intenta predecir.
df = pd.read_csv('/content/drive/MyDrive/OctavoSemestre_Ago23_ConcentraciónIA/M2-Uresti/Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = df.drop('Performance Index', axis=1).values
y = df['Performance Index'].values

#División del conjunto de datos en conjuntos de entrenamiento y prueba:
#Se divide el conjunto de datos en tres partes: entrenamiento (80%), validación (20%) y prueba (20%). 
#Esto se hace utilizando la función train_test_split dos veces. Primero, se divide el conjunto de datos en entrenamiento (80%) 
#y prueba+validación (20%). Luego, se divide nuevamente el conjunto de entrenamiento en entrenamiento (80%) y validación (20%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

#Creación y entrenamiento del modelo KNN
#Se crea un modelo de regresión KNN (KNeighborsRegressor) con 5 vecinos y otros parámetros predeterminados. Luego, ajusta el modelo 
#a los datos de entrenamiento (x_train e y_train). Los hiperparámetros que se están configurando en esta instancia son los siguientes:
# n_neighbors: Este hiperparámetro controla el número de vecinos más cercanos que se utilizarán para hacer predicciones. En este caso, se 
#ha establecido en 5, lo que significa que el modelo KNN considerará los 5 vecinos más cercanos para hacer una predicción para cada punto de datos nuevo.
#weights: Este hiperparámetro determina cómo se ponderan los vecinos cercanos cuando se realiza una predicción. Toma 'uniform', es decir, 
#todos los vecinos se ponderan igualmente, lo que significa que tienen el mismo impacto en la predicción.
#algorithm: Este hiperparámetro controla el algoritmo utilizado para calcular los vecinos más cercanos. Considera 'auto', donde deja que el 
#modelo elija automáticamente el algoritmo más apropiado en función de los datos de entrada y otros factores.
knn_regr = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
knn_regr.fit(x_train, y_train)

#Predicción y evaluación en el conjunto de validación:
#Se utiliza el modelo entrenado para predecir los valores del conjunto de validación (x_val) y calcula varias métricas de evaluación, 
#como el Error Absoluto Medio (MAE), el Error Cuadrático Medio (MSE), la Raíz del Error Cuadrático Medio (RMSE) y el coeficiente de determinación 
#(R2) en el conjunto de validación.
y_pred = knn_regr.predict(x_val)
predictions = pd.DataFrame({'Actual': y_val, 'Predicted': y_pred})
print("Validation Set Predictions:")
print(predictions)

mae = mean_absolute_error(y_val, y_pred)
print("Validation Set MAE:", mae)
mse = mean_squared_error(y_val, y_pred)
print("Validation Set MSE:", mse)
rmse = np.sqrt(mse)
print("Validation Set RMSE:", rmse)
r2 = r2_score(y_val, y_pred)
print("Validation Set R2:", r2)

#Predicción y evaluación en el conjunto de prueba
#Se utiliza el modelo entrenado para predecir los valores del conjunto de prueba (x_test) y calcula las mismas 
#métricas de evaluación que en el paso anterior, pero esta vez en el conjunto de prueba.
y_pred_test = knn_regr.predict(x_test)
predictions_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
print("\nTest Set Predictions:")
print(predictions_test)

mae_test = mean_absolute_error(y_test, y_pred_test)
print("Test Set MAE:", mae_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test Set MSE:", mse_test)
rmse_test = np.sqrt(mse_test)
print("Test Set RMSE:", rmse_test)
r2_test = r2_score(y_test, y_pred_test)
print("Test Set R2:", r2_test)
