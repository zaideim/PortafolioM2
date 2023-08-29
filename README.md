# PortafolioM2

## Instrucciones
Entregable: Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

1. Crea un repositorio de GitHub para este proyecto.
2. Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) sin usar ninguna biblioteca o framework de aprendizaje máquina, ni de estadística avanzada. Lo que se busca es que implementes manualmente el algoritmo, no que importes un algoritmo ya implementado. 
3. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
4. Tu implementación debe de poder correr por separado solamente con un compilador, no debe de depender de un IDE o de un “notebook”. Por ejemplo, si programas en Python, tu implementación final se espera que esté en un archivo .py no en un Jupyter Notebook.
5. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

## Entregable
### 1. Acerca del Dataset:
El Student Performance Dataset es un conjunto de datos diseñado para examinar los factores que influyen en el rendimiento académico de los estudiantes. El conjunto de datos consta de 10.000 registros de estudiantes, cada uno de los cuales contiene información sobre diversos predictores y un índice de rendimiento.

**Variables:**
*  Horas estudiadas: El número total de horas dedicadas al estudio por cada estudiante.
*  Puntuaciones anteriores: Las puntuaciones obtenidas por los alumnos en pruebas anteriores.
*  Actividades Extraescolares: Si el alumno participa en actividades extraescolares (Sí o No).
*  Horas de sueño: El número medio de horas de sueño diarias del alumno.
*  Cuestionarios de muestra practicados: Número de cuestionarios de muestra que ha practicado el alumno.
*  Variable objetivo - Índice de rendimiento: Medida del rendimiento global de cada alumno. El índice de rendimiento representa el rendimiento académico del alumno y se ha redondeado al número entero más próximo. El índice oscila entre 10 y 100, y los valores más altos indican un mejor rendimiento.
El objetivo del conjunto de datos es proporcionar información sobre la relación entre las variables predictoras y el índice de rendimiento. Los investigadores y analistas de datos pueden utilizar este conjunto de datos para explorar el impacto de las horas de estudio, las calificaciones anteriores, las actividades extraescolares, las horas de sueño y los modelos de cuestionarios en el rendimiento de los estudiantes.

*Conjunto de datos obtenido de Narayan, N. (2023). Student Performance [Data set]. En Student Performance (Multiple Linear Regression). https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

### 2. Acerca del modelo:
k-Nearest Neighbors (kNN) es un algoritmo de aprendizaje automático supervisado utilizado tanto para clasificación como para regresión. Su enfoque es simple pero efectivo: clasifica o predice nuevas instancias en función de las instancias existentes en el conjunto de datos de entrenamiento más cercanas en términos de distancia.

En la clasificación kNN, cuando se le presenta una nueva instancia, el algoritmo busca las k instancias más cercanas en el conjunto de entrenamiento y determina la clase mayoritaria entre esas k instancias. La instancia se clasifica en esa clase mayoritaria. La elección del valor de k es crucial, ya que un valor pequeño puede hacer que el algoritmo sea muy sensible al ruido en los datos, mientras que un valor grande puede diluir los patrones subyacentes. En la regresión kNN, el algoritmo predice el valor de una variable numérica para una nueva instancia al promediar los valores de las k instancias más cercanas en el conjunto de entrenamiento.

El proceso de cálculo de la distancia entre instancias puede variar, pero la distancia euclidiana es la más común. Otras métricas de distancia, como la distancia de Manhattan o la distancia de Minkowski, también pueden utilizarse según el contexto.

Una de las ventajas del algoritmo kNN es su simplicidad y capacidad para capturar relaciones no lineales en los datos. Sin embargo, también tiene limitaciones. Puede ser computacionalmente costoso en conjuntos de datos grandes, y su rendimiento puede verse afectado por la presencia de características irrelevantes o redundantes. Además, el rendimiento puede verse influenciado por la elección adecuada de la métrica de distancia y el valor de k.

El código es una implementación de "k-Nearest Neighbors" sin el uso de un framework de ML.
1. Importa las bibliotecas necesarias: csv, math, pandas, matplotlib.pyplot, train_test_split de sklearn.model_selection, KNeighborsClassifier de sklearn.neighbors y accuracy_score de sklearn.metrics.
2. Define dos funciones: euclidean_distance(point1, point2) para calcular la distancia euclidiana entre dos puntos y predict_knn(training_data, test_point, k) para realizar una predicción utilizando el algoritmo de los k-vecinos más cercanos (KNN).}
3. Lee los datos relacionados con el rendimiento de estudiantes. Los datos se cargan en una lista llamada data después de realizar algunas transformaciones en los valores. Divide los datos en conjuntos de entrenamiento y prueba utilizando la función train_test_split. El 70% de los datos se utiliza como conjunto de prueba y el 30% como conjunto de prueba. El parámetro random_state se establece en 42 para asegurar la reproducibilidad de la división.
4. Realiza una búsqueda de hiperparámetros para encontrar el valor óptimo de k en el algoritmo KNN. Itera a través de valores de k desde 1 hasta 199. Para cada valor de k, evalúa la precisión promedio en el conjunto de prueba utilizando el algoritmo KNN. La precisión se calcula comparando las etiquetas predichas con las etiquetas verdaderas y contando el número de predicciones correctas. El valor de k que produce la mayor precisión promedio se almacena en best_k. Utiliza el valor óptimo de k para realizar predicciones finales en el conjunto de prueba. Para cada punto en el conjunto de prueba, utiliza el algoritmo KNN para predecir su etiqueta. Luego, compara la etiqueta predicha con la etiqueta verdadera y calcula la precisión individual. Las predicciones y precisiones individuales se imprimen en pantalla.
5. Calcula la precisión promedio final en el conjunto de prueba utilizando el valor óptimo de k. Esto se hace sumando todas las precisiones individuales y dividiendo entre el número total de puntos en el conjunto de prueba.
