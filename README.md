# PortafolioM2

# Entregable 1: Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

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

# Entregable 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución. 

## Instrucciones
Entregable: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.

1. Crea un repositorio de GitHub para este proyecto.
2. Programa uno de los algoritmos vistos en el módulo (o que tu profesor de módulo autorice) haciendo uso de una biblioteca o framework de aprendizaje máquina. Lo que se busca es que demuestres tu conocimiento sobre el framework y como configurar el algoritmo. 
3. Prueba tu implementación con un set de datos y realiza algunas predicciones. Las predicciones las puedes correr en consola o las puedes implementar con una interfaz gráfica apoyándote en los visto en otros módulos.
4. Tu implementación debe de poder correr por separado solamente con un compilador, no debe de depender de un IDE o de un “notebook”. Por ejemplo, si programas en Python, tu implementación final se espera que esté en un archivo .py no en un Jupyter Notebook.
5. Después de la entrega intermedia se te darán correcciones que puedes incluir en tu entrega final.

## Entregable
### 1. Acerca del Dataset:
El dataset utilizado es el mismo del entregable anterior. En lo personal considero que es de buen uso, pues contiene una cantidad muy considerable de datos, lo que hace que los modelos sean implementados de mejor forma, además de que no necesita pre-procesamiento de datos. 

### 2. KNearest Neighbors:
K-Nearest Neighbors (KNN), que se traduce como "K Vecinos Más Cercanos" en español, es un algoritmo de aprendizaje supervisado utilizado tanto en problemas de clasificación como de regresión. Su idea fundamental es bastante simple:

1. Vecindad basada en la distancia: KNN se basa en la suposición de que los puntos de datos similares deben estar cerca en el espacio de características. Para realizar predicciones, calcula la distancia (por ejemplo, distancia euclidiana) entre el punto de datos que se quiere clasificar o predecir y todos los demás puntos de datos en el conjunto de entrenamiento.

2. Votación o promedio ponderado: Una vez que se han calculado las distancias, KNN selecciona los K puntos de datos más cercanos al punto de interés. En el caso de clasificación, estos puntos "votan" para determinar la etiqueta de clasificación del punto de interés. En el caso de regresión, estos puntos contribuyen con sus valores para calcular un promedio ponderado como predicción.

Algunos aspectos clave de KNN:

- Hiperparámetro K: El valor de K es uno de los hiperparámetros más importantes en KNN. Controla la cantidad de vecinos que se considerarán al tomar una decisión. Un K pequeño (por ejemplo, K = 1) puede hacer que el modelo sea muy sensible al ruido, mientras que un K grande puede suavizar demasiado las fronteras de decisión.

- Función de distancia: La elección de la función de distancia, como la distancia euclidiana, Manhattan u otras métricas, puede influir en el rendimiento del modelo y su capacidad para manejar diferentes tipos de datos.

- Ponderación de vecinos: En el caso de regresión, se puede ponderar a los vecinos en función de su distancia al punto de interés. Esto significa que los vecinos más cercanos pueden tener un impacto mayor en la predicción que los vecinos más alejados.

- Escalado de características: Es importante escalar las características antes de aplicar KNN, ya que las diferencias en las escalas de las características pueden tener un impacto desproporcionado en la distancia y, por lo tanto, en las predicciones.

KNN es un algoritmo simple pero efectivo, especialmente cuando se tiene poca cantidad de datos o no se dispone de información suficiente sobre la distribución de los datos. Sin embargo, su rendimiento puede verse afectado por la elección adecuada de K y la función de distancia, y puede no funcionar bien en dimensiones muy altas debido al "problema de la maldición de la dimensionalidad".

### 3. Acerca de la implementación del modelo:

1. Importación de bibliotecas: 
 - Se importa las bibliotecas necesarias, incluyendo Pandas, NumPy y varias funciones y clases relacionadas con la selección de modelos y métricas de regresión de scikit-learn.
2. Lectura de datos:
 - Se lee un conjunto de datos desde un archivo CSV llamado "Student_Performance.csv".
3. Conversión de valores categóricos:
 - Se convierte los valores categóricos en la columna "Extracurricular Activities" de "Yes" a 1 y de "No" a 0. Esto es necesario para que el algoritmo KNN pueda manejar estos valores categóricos como numéricos.
4. Separación de características y variable objetivo:
 - Se divide el conjunto de datos en dos partes:
    - X: Contiene todas las características excepto la columna "Performance Index".
    - y: Contiene la columna "Performance Index", que es la variable objetivo que se intenta predecir.
5. División del conjunto de datos en conjuntos de entrenamiento y prueba:
  - Se divide el conjunto de datos en tres partes: entrenamiento (80%), validación (20%) y prueba (20%). Esto se hace utilizando la función train_test_split dos veces. Primero, se divide el conjunto de datos en entrenamiento (80%) y prueba+validación (20%). Luego, se divide nuevamente el conjunto de entrenamiento en entrenamiento (80%) y validación (20%).
6. Creación y entrenamiento del modelo KNN:
  - Se crea un modelo de regresión KNN (KNeighborsRegressor) con 5 vecinos y otros parámetros predeterminados. Luego, ajusta el modelo a los datos de entrenamiento (x_train e y_train). Los hiperparámetros que se están configurando en esta instancia son los siguientes:
    * n_neighbors: Este hiperparámetro controla el número de vecinos más cercanos que se utilizarán para hacer predicciones. En este caso, se ha establecido en 5, lo que significa que el modelo KNN considerará los 5 vecinos más cercanos para hacer una predicción para cada punto de datos nuevo.
    * weights: Este hiperparámetro determina cómo se ponderan los vecinos cercanos cuando se realiza una predicción. Toma 'uniform', es decir, todos los vecinos se ponderan igualmente, lo que significa que tienen el mismo impacto en la predicción.
    *  algorithm: Este hiperparámetro controla el algoritmo utilizado para calcular los vecinos más cercanos. Considera 'auto', donde deja que el modelo elija automáticamente el algoritmo más apropiado en función de los datos de entrada y otros factores.
7. Predicción y evaluación en el conjunto de validación:
  - Se utiliza el modelo entrenado para predecir los valores del conjunto de validación (x_val) y calcula varias métricas de evaluación, como el Error Absoluto Medio (MAE), el Error Cuadrático Medio (MSE), la Raíz del Error Cuadrático Medio (RMSE) y el coeficiente de determinación (R2) en el conjunto de validación.
8. Predicción y evaluación en el conjunto de prueba:
  - Se utiliza el modelo entrenado para predecir los valores del conjunto de prueba (x_test) y calcula las mismas métricas de evaluación que en el paso anterior, pero esta vez en el conjunto de prueba.
9. Imprime los resultados:
  - Se imprime las predicciones y las métricas de evaluación tanto para el conjunto de validación como para el conjunto de prueba.

### 4. Evaluación del modelo:

#### Uso de métricas
1. MAE (Mean Absolute Error - Error Absoluto Medio): El MAE es una medida de la magnitud promedio de los errores en las predicciones del modelo. Cuanto menor sea el MAE, mejor será el modelo en términos de precisión.

2. MSE (Mean Squared Error - Error Cuadrático Medio): El MSE mide la magnitud promedio de los errores al cuadrado. Es más sensible a errores grandes debido a la naturaleza de elevar al cuadrado. Cuanto menor sea el MSE, mejor será el modelo en términos de precisión.

3. RMSE (Root Mean Squared Error - Raíz del Error Cuadrático Medio): El RMSE es simplemente la raíz cuadrada del MSE y se expresa en la misma unidad que la variable objetivo. Es una medida útil para comprender el error promedio de las predicciones en la misma escala que la variable objetivo.  Al igual que el MAE y el MSE, un valor más bajo de RMSE indica un mejor rendimiento del modelo.

4. R2 (R-cuadrado): El coeficiente de determinación R2 es una medida que indica la proporción de la variabilidad total en la variable objetivo que es explicada por el modelo. Los valores de R2 varían entre 0 y 1, donde 1 indica una ajuste perfecto del modelo a los datos y 0 indica que el modelo no explica nada de la variabilidad en los datos.

#### Interpretación de resultados
Los resultados impresos muestran las métricas de evaluación del modelo de regresión K Vecinos más Cercanos (KNN) tanto en el conjunto de validación como en el conjunto de prueba.

**Conjunto de Validación:**
*  Validation Set MAE: En este caso, el MAE es aproximadamente 1.9465. Cuanto más bajo sea el MAE, mejor será el rendimiento del modelo.
*  Validation Set MSE:En este caso, el MSE es aproximadamente 6.0927. Al igual que el MAE, un MSE más bajo es mejor.
* Validation Set RMSE: En este caso, el RMSE es aproximadamente 2.4683.
* Validation Set R2 (Coeficiente de Determinación): En este caso, R2 es aproximadamente 0.9835, lo que significa que el modelo explica el 98.35% de la varianza en el "Performance Index" en el conjunto de validación. Un valor cercano a 1 indica un buen ajuste del modelo.

**Conjunto de Prueba:**
* Test Set MAE: El MAE en el conjunto de prueba es aproximadamente 1.9642.
* Test Set MSE: El MSE en el conjunto de prueba es aproximadamente 6.0792.
* Test Set RMSE: El RMSE en el conjunto de prueba es aproximadamente 2.4656.
* Test Set R2: El R2 en el conjunto de prueba es aproximadamente 0.9836.

En resumen, estos resultados indican que el modelo KNN tiene un buen rendimiento tanto en el conjunto de validación como en el conjunto de prueba. Las métricas de evaluación (MAE, MSE, RMSE y R2) son muy similares en ambos conjuntos, lo que sugiere que el modelo generaliza bien y no muestra signos de sobreajuste. El alto valor de R2 cercano a 1 en ambos conjuntos indica que el modelo explica la mayor parte de la variabilidad en el "Performance Index".
