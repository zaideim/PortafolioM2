#Librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Lectura de datos
df = pd.read_csv('/content/drive/MyDrive/OctavoSemestre_Ago23_ConcentraciónIA/M2-Uresti/Student_Performance.csv')
# Convertir valores categóricos en numéricos (Yes=1, No=0)
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
# Separar las características (X) y la variable objetivo (y)
X = df.drop('Performance Index', axis=1).values
y = df['Performance Index'].values

# Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Dividir el conjunto de entrenamiento en entrenamiento y validación (80% entrenamiento, 20% validación)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

knn_regr = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
knn_regr.fit(x_train, y_train)

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

# Ahora, evaluamos en el conjunto de prueba
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
