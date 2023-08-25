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

# Distancia euclidiana entre dos puntos
def euclidean_distance(point1, point2):
    squared_distance = 0
    for i in range(len(point1)):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)

# Predecir la etiqueta de un punto basado en sus kNN
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

training_data = []
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
        
        training_data.append([hours_studied, previous_scores, extracurricular, sleep_hours, question_papers, performance_index])

# Datos de prueba (Horas Estudiadas, Puntajes Anteriores, Actividades Extraescolares, Horas de Sueño, Preguntas Practicadas)
test_data = [
    [6, 80, 1, 6, 3],
    [9, 45, 0, 7, 1],
    [5, 30, 1, 5, 5],
    [2, 95, 0, 0, 2],
    [0, 56, 1, 3, 3]
]

k = 3  

# Predicciones para datos de prueba
for test_point in test_data:
    predicted_label = predict_knn(training_data, test_point, k)
    print(f"Para el punto {test_point}, la predicción es: {predicted_label}")
