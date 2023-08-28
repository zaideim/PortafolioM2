# PortafolioM2

## Instrucciones:
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

Este código es una implementación de un algoritmo de Machine Learning llamado "k-Nearest Neighbors" (k-NN) sin el uso de un framework de ML.
1. Importación de bibliotecas: El código comienza importando las bibliotecas necesarias para realizar el análisis y la implementación del algoritmo. Esto incluye la importación de las bibliotecas `csv`, `math`, `pandas`, `matplotlib.pyplot` y `sklearn.model_selection`. Estas bibliotecas se utilizarán para leer datos desde un archivo CSV, realizar cálculos matemáticos, visualizar datos y dividir los datos en conjuntos de entrenamiento y prueba.
2. Definición de funciones:
   - `euclidean_distance`: Esta función calcula la distancia euclidiana entre dos puntos en un espacio n-dimensional. Se utiliza para medir la similitud entre los datos.
   - `predict_knn`: Esta función implementa el algoritmo k-NN para predecir la etiqueta de un punto de prueba basado en sus k vecinos más cercanos en el conjunto de entrenamiento. Utiliza la distancia euclidiana para encontrar los vecinos más cercanos y luego realiza una votación para determinar la etiqueta más probable.
3. Exploración de datos: Se han comentado algunas líneas de código que, si se descomentan, permitirían la exploración de los datos descritos previamente. Estas líneas verificarían la presencia de valores faltantes, duplicados y trazarían un gráfico de caja para detectar valores atípicos en los datos.
4. Entrenamiento con datos:
   - Se carga el conjunto de datos desde el archivo CSV 'Student_Performance.csv' y se almacena en la lista `data`.
   - Los valores relevantes se extraen de cada fila del archivo CSV y se almacenan en la lista `data` después de realizar algunas conversiones de tipo.
   - Los datos se dividen en conjuntos de entrenamiento y prueba utilizando la función `train_test_split` de `sklearn`. El 80% de los datos se utiliza para entrenar el modelo y el 20% se utiliza para realizar predicciones y evaluar el rendimiento del modelo.
5. Predicciones:
   - Se define el valor de `k` como 3, lo que significa que se considerarán los 3 vecinos más cercanos para cada punto de prueba.
   - Se itera a través de los puntos de prueba en `test_data` y se utiliza la función `predict_knn` para predecir la etiqueta de cada punto de prueba basándose en sus vecinos más cercanos en el conjunto de entrenamiento.
   - Se imprime la predicción para cada punto de prueba en el formato "Para el punto [datos del punto], la predicción es: [etiqueta predicha]".
