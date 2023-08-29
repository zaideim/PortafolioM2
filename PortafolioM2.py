# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:12:55 2023

@author: zayde
"""

# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
#Zaide Islas Montiel A01751580
#28/Agosto/2023

import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Modelo KNN
# Distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    squared_distance = 0
    for i in range(len(point1)):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)

# Vecinos más cercanos
def predict_knn(training_data, test_point, k):
    distances = []
    for train_point in training_data:
        distance = euclidean_distance(train_point[:-1], test_point)
        distances.append((train_point, distance))
    distances.sort(key=lambda x: x[1])

    neighbors = [item[0] for item in distances[:k]]
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

#Exploración de datos
#df = pd.read_csv('Student_Performance.csv')
#print(df.isnull().sum()) #No hay datos faltantes
#df.duplicated() #No hay valores duplicados
#No hay outliers
#fig, ax = plt.subplots(figsize=(10,10))
#df.boxplot(ax=ax)
#plt.xticks(rotation=90)
#plt.show()

# Entrenamiento con datos
data = []
with open('Student_Performance.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        hours_studied = int(row[0])
        previous_scores = int(row[1])
        extracurricular = 1 if row[2] == 'Yes' else 0
        sleep_hours = int(row[3])
        question_papers = int(row[4])
        performance_index = float(row[5])
        data.append([hours_studied, previous_scores, extracurricular, sleep_hours, question_papers, performance_index])

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Búsqueda de hiperparámetros para encontrar la cantidad correcta de vecinos
best_k = None
best_accuracy = 0.0

for k_candidate in range(1,200):
    current_accuracies = []

    for test_point in test_data:
        predicted_label = predict_knn(train_data, test_point, k_candidate)
        true_label = test_point[-1]
        accuracy = 1 if predicted_label == true_label else 0
        current_accuracies.append(accuracy)

    current_average_accuracy = sum(current_accuracies) / len(current_accuracies)

    if current_average_accuracy > best_accuracy:
        best_accuracy = current_average_accuracy
        best_k = k_candidate

    print(f"Para k = {k_candidate}, la precisión promedio es: {current_average_accuracy}")

print(f"El valor de k que produce la mayor precisión promedio es: {best_k}")

# Usar el valor óptimo de k para las predicciones finales
final_accuracies = []

for test_point in test_data:
    predicted_label = predict_knn(train_data, test_point, best_k)
    true_label = test_point[-1]
    accuracy = 1 if predicted_label == true_label else 0
    final_accuracies.append(accuracy)
    print(f"Para el punto {test_point}, la predicción es: {predicted_label}")

final_average_accuracy = sum(final_accuracies) / len(final_accuracies)
print(f"Precisión promedio final: {final_average_accuracy}")
