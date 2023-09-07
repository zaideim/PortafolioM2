# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:12:55 2023

@author: zayde
"""

#Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
#Zaide Islas Montiel A01751580
#28/Agosto/2023

#Importación de bibliotecas
#Estas bibliotecas se utilizan para realizar cálculos numéricos y para la manipulación de datos.
import numpy as np
import pandas as pd

# Entropía
#Esta función calcula la entropía de un conjunto de datos etiquetados y. La entropía es una medida de la 
#impureza de un conjunto de datos y se utiliza en la construcción de árboles de decisión. La fórmula de la 
#entropía se utiliza para calcular la incertidumbre en los datos.
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Ganancia de información
#Esta función calcula la ganancia de información al dividir un conjunto de datos y en varios subconjuntos y_split. 
#La ganancia de información se utiliza para seleccionar la mejor característica para dividir en un árbol de decisión. 
#Cuanto mayor sea la ganancia de información, mejor será la división.
def information_gain(y, y_split):
    entropy_before = entropy(y)
    entropy_after = sum((len(y_split[i]) / len(y)) * entropy(y_split[i]) for i in range(len(y_split)))
    return entropy_before - entropy_after

#Nodo del árbol de decisión
#Esta clase representa un nodo en el árbol de decisión. Cada nodo tiene un índice de característica (feature_index), 
#un umbral (threshold), un valor (value), un nodo verdadero (true_branch) y un nodo falso (false_branch). Los nodos hoja 
#tienen un valor asignado, mientras que los nodos internos tienen un índice de característica y un umbral para tomar decisiones.
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

#Construir el árbol de decisión
#Esta función construye un árbol de decisión recursivamente utilizando el algoritmo de selección de la mejor característica y división. 
#Toma como entrada un conjunto de datos X y las etiquetas correspondientes y, así como los hiperparámetros max_depth (profundidad máxima del árbol) y 
#min_samples_split (número mínimo de muestras para dividir un nodo). El árbol se construye de manera recursiva, dividiendo los nodos en función de la ganancia de información.
def build_decision_tree(X, y, max_depth=None, min_samples_split=2):
    num_samples, num_features = X.shape
    unique_labels = np.unique(y)

    # Si todos los ejemplos pertenecen a la misma clase o hemos alcanzado la profundidad máxima, devolvemos un nodo hoja
    if len(unique_labels) == 1 or (max_depth and max_depth == 0) or (num_samples < min_samples_split):
        return DecisionNode(value=unique_labels[0])

    # Encontrar la mejor división
    best_feature_index = None
    best_threshold = None
    best_info_gain = -1

    for feature_index in range(num_features):
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)

        for threshold in unique_values:
            left_indices = X[:, feature_index] <= threshold
            right_indices = X[:, feature_index] > threshold
            current_info_gain = information_gain(y, [y[left_indices], y[right_indices]])

            if current_info_gain > best_info_gain:
                best_feature_index = feature_index
                best_threshold = threshold
                best_info_gain = current_info_gain

    if best_info_gain == 0:
        return DecisionNode(value=unique_labels[0])

    # Dividir el conjunto de datos
    left_indices = X[:, best_feature_index] <= best_threshold
    right_indices = X[:, best_feature_index] > best_threshold

    true_branch = build_decision_tree(X[left_indices], y[left_indices], max_depth=max_depth - 1 if max_depth else None, min_samples_split=min_samples_split)
    false_branch = build_decision_tree(X[right_indices], y[right_indices], max_depth=max_depth - 1 if max_depth else None, min_samples_split=min_samples_split)

    return DecisionNode(feature_index=best_feature_index, threshold=best_threshold, true_branch=true_branch, false_branch=false_branch)

# Define la función para realizar predicciones
def predict_tree(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature_index] <= node.threshold:
        return predict_tree(node.true_branch, X)
    else:
        return predict_tree(node.false_branch, X)

# Carga los datos
#Se carga un conjunto de datos desde un archivo CSV llamado 'Student_Performance.csv' en un DataFrame de pandas (df). Luego, 
#la columna 'Extracurricular Activities' se mapea de 'Yes' y 'No' a 1 y 0, respectivamente. Las características se almacenan en X, y la 
#variable objetivo se almacena en y.
df = pd.read_csv('Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
X = df.drop('Performance Index', axis=1).values
y = df['Performance Index'].values

# Entrenar el árbol de decisión
#Se construye el árbol de decisión llamando a build_decision_tree con los datos de entrenamiento X y y. Se 
#especifica una profundidad máxima de 5 niveles (max_depth=5) y un número mínimo de muestras para dividir un nodo de 2 (min_samples_split=2).
tree = build_decision_tree(X, y, max_depth=5, min_samples_split=2)

#Se utilizan las predicciones del árbol de decisión para el mismo conjunto de datos de entrenamiento. Cada instancia de X se pasa a la 
#función predict_tree, que recorre el árbol y devuelve una predicción para esa instancia.
predictions = [predict_tree(tree, x) for x in X]

# Calcular precisión, exactitud, recall y f1-score
#Se calculan métricas de evaluación del modelo, como la precisión (accuracy), la precisión ponderada (precision), la recuperación 
#ponderada (recall) y la puntuación F1 ponderada (f1) utilizando las funciones de la biblioteca scikit-learn 
#(accuracy_score, precision_score, recall_score, f1_score). Estas métricas evalúan el rendimiento del modelo en las predicciones 
#realizadas en comparación con las etiquetas reales.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average='weighted')
recall = recall_score(y, predictions, average='weighted')
f1 = f1_score(y, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

"""
Resultados de diferentes corridas

Corrida 1: max_depth=5, min_samples_split=3
Accuracy: 0.755
Precision: 0.7564911014165513
Recall: 0.755
F1 Score: 0.7545749508693737

Corrida 2: max_depth=10, min_samples_split=5
Accuracy: 0.5116
Precision: 0.5132543689517636
Recall: 0.5116
F1 Score: 0.509642403097133

Corrida 3: max_depth=5, min_samples_split=2
Accuracy: 0.9315
Precision: 0.9321523558106047
Recall: 0.9315
F1 Score: 0.9314790442752705
"""
