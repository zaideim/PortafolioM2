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

### 2. Árboles de decisión:
Un árbol de decisión es una representación gráfica de un proceso de toma de decisiones que se utiliza en una variedad de campos, como la inteligencia artificial, la estadística, la minería de datos y la toma de decisiones empresariales. Se utiliza para modelar decisiones y sus posibles consecuencias en forma de una estructura jerárquica similar a un árbol, donde cada nodo representa una decisión o una prueba sobre un atributo, cada rama representa un resultado posible de la decisión o prueba, y cada hoja del árbol representa una decisión final o un resultado.

#### **Componentes clave**

1. **Nodo Raíz:** El nodo superior del árbol que representa la decisión inicial o la pregunta inicial. Este nodo se divide en ramas que representan diferentes opciones o pruebas disponibles.

2. **Nodos Internos:** Los nodos intermedios en el árbol que representan decisiones o pruebas adicionales basadas en las opciones anteriores. Cada nodo interno se conecta a uno o más nodos secundarios a través de ramas.

3. **Ramas:** Las aristas o líneas que conectan los nodos y representan las diferentes opciones o resultados de una decisión o prueba. Cada rama se etiqueta con la opción o el resultado correspondiente.

4. **Nodos Hoja:** Los nodos finales del árbol que representan las decisiones finales o los resultados de todo el proceso de toma de decisiones. Estos nodos no tienen ramas salientes y proporcionan la respuesta final o la clasificación.

5. **Atributos:** Los atributos son las características o variables que se utilizan para tomar decisiones en cada nodo interno. Por ejemplo, en un árbol de decisión para predecir si un correo electrónico es spam o no, los atributos podrían incluir palabras clave en el correo electrónico, la dirección del remitente, etc.

6. **Criterio de División:** En cada nodo interno, se utiliza un criterio de división para decidir cómo se separan las opciones en las ramas salientes. Los criterios comunes incluyen la ganancia de información, la impureza de Gini y la entropía.

7. **P poda:** La poda es un proceso opcional que implica la eliminación de ciertas ramas o nodos para simplificar el árbol y evitar el sobreajuste (cuando el árbol se adapta demasiado a los datos de entrenamiento y no generaliza bien).

8. **Clasificación o Predicción:** Un árbol de decisión puede utilizarse para clasificar datos o hacer predicciones. En el caso de clasificación, se asigna una etiqueta o categoría a un objeto o una observación, mientras que en la predicción se estima un valor numérico.

El proceso de construcción de un árbol de decisión implica dividir el conjunto de datos de entrenamiento en función de los atributos y el criterio de división seleccionado de manera iterativa, de manera que se maximice la homogeneidad o se minimice la impureza en las ramas resultantes. Este proceso continúa hasta que se alcanza un criterio de detención predefinido, como la profundidad máxima del árbol o la impureza mínima.

Los árboles de decisión son útiles en la toma de decisiones, la clasificación y la predicción debido a su capacidad para representar relaciones complejas entre variables y para generar reglas interpretables. Sin embargo, pueden ser propensos al sobreajuste si se construyen sin restricciones, por lo que es importante ajustar parámetros y realizar la poda adecuada para obtener modelos más generalizables. Además, se utilizan en conjunto con técnicas como el conjunto de árboles (Random Forests) y el aumento de gradiente (Gradient Boosting) para mejorar su rendimiento.

### 3. Acerca de la implementación del modelo:
1. Importación de bibliotecas:
   - Se importan las bibliotecas `numpy` y `pandas` como `np` y `pd`, respectivamente. Estas bibliotecas se utilizan para realizar cálculos numéricos y para la manipulación de datos.

2. Definición de la función `entropy(y)`:
   - Esta función calcula la entropía de un conjunto de datos etiquetados `y`. La entropía es una medida de la impureza de un conjunto de datos y se utiliza en la construcción de árboles de decisión. La fórmula de la entropía se utiliza para calcular la incertidumbre en los datos.

3. Definición de la función `information_gain(y, y_split)`:
   - Esta función calcula la ganancia de información al dividir un conjunto de datos `y` en varios subconjuntos `y_split`. La ganancia de información se utiliza para seleccionar la mejor característica para dividir en un árbol de decisión. Cuanto mayor sea la ganancia de información, mejor será la división.

4. Definición de la clase `DecisionNode`:
   - Esta clase representa un nodo en el árbol de decisión. Cada nodo tiene un índice de característica (`feature_index`), un umbral (`threshold`), un valor (`value`), un nodo verdadero (`true_branch`) y un nodo falso (`false_branch`). Los nodos hoja tienen un valor asignado, mientras que los nodos internos tienen un índice de característica y un umbral para tomar decisiones.

5. Definición de la función `build_decision_tree(X, y, max_depth=None, min_samples_split=2)`:
   - Esta función construye un árbol de decisión recursivamente utilizando el algoritmo de selección de la mejor característica y división. Toma como entrada un conjunto de datos `X` y las etiquetas correspondientes `y`, así como los hiperparámetros `max_depth` (profundidad máxima del árbol) y `min_samples_split` (número mínimo de muestras para dividir un nodo). El árbol se construye de manera recursiva, dividiendo los nodos en función de la ganancia de información.

6. Lectura y preprocesamiento de datos:
   - Se carga un conjunto de datos desde un archivo CSV llamado 'Student_Performance.csv' en un DataFrame de pandas (`df`). Luego, la columna 'Extracurricular Activities' se mapea de 'Yes' y 'No' a 1 y 0, respectivamente. Las características se almacenan en `X`, y la variable objetivo se almacena en `y`.

7. Entrenamiento del árbol de decisión:
   - Se construye el árbol de decisión llamando a `build_decision_tree` con los datos de entrenamiento `X` y `y`. Se especifica una profundidad máxima de 5 niveles (`max_depth=5`) y un número mínimo de muestras para dividir un nodo de 2 (`min_samples_split=2`).

8. Realización de predicciones:
   - Se utilizan las predicciones del árbol de decisión para el mismo conjunto de datos de entrenamiento. Cada instancia de `X` se pasa a la función `predict_tree`, que recorre el árbol y devuelve una predicción para esa instancia.

9. Evaluación del modelo:
   - Se calculan métricas de evaluación del modelo, como la precisión (`accuracy`), la precisión ponderada (`precision`), la recuperación ponderada (`recall`) y la puntuación F1 ponderada (`f1`) utilizando las funciones de la biblioteca `scikit-learn` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`). Estas métricas evalúan el rendimiento del modelo en las predicciones realizadas en comparación con las etiquetas reales.

10. Impresión de resultados:
   - Se imprimen las métricas de evaluación (precisión, precisión ponderada, recuperación ponderada y puntuación F1 ponderada) en la consola.
