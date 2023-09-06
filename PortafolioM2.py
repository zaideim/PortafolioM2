# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:12:55 2023

@author: zayde
"""

#Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)
#Zaide Islas Montiel A01751580
#28/Agosto/2023

import numpy as np
import pandas as pd

#Entropía
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

#Ganancia de información
def information_gain(y, y_split):
    entropy_before = entropy(y)
    entropy_after = sum((len(y_split[i]) / len(y)) * entropy(y_split[i]) for i in range(len(y_split)))
    return entropy_before - entropy_after

#Nodo del árbol de decisión
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

#Construir el árbol de decisión
def build_decision_tree(X, y, max_depth=None, min_samples_split=2):
    num_samples, num_features = X.shape
    unique_labels = np.unique(y)
    if len(unique_labels) == 1 or (max_depth and max_depth == 0) or (num_samples < min_samples_split):
        return DecisionNode(value=unique_labels[0])
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

#Predicciones
def predict_tree(node, X):
    if node.value is not None:
        return node.value
    if X[node.feature_index] <= node.threshold:
        return predict_tree(node.true_branch, X)
    else:
        return predict_tree(node.false_branch, X)

df = pd.read_csv('Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Separar las características (X) y la variable objetivo (y)
X = df.drop('Performance Index', axis=1).values
y = df['Performance Index'].values

# Entrenar el árbol de decisión
tree = build_decision_tree(X, y, max_depth=5, min_samples_split=2)

# Realizar predicciones en el mismo conjunto de datos de entrenamiento
predictions = [predict_tree(tree, x) for x in X]

# Calcular precisión, exactitud, recall y f1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average='weighted')
recall = recall_score(y, predictions, average='weighted')
f1 = f1_score(y, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
