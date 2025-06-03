# Proyecto: Detección de Fraude en Transacciones con Tarjeta de Crédito

## Descripción del problema
Este proyecto tiene como objetivo predecir transacciones fraudulentas en un histórico de tarjetas de crédito utilizando técnicas de Machine Learning. A partir de un conjunto de datos etiquetado con transacciones legítimas y fraudulentas, se implementaron y compararon dos algoritmos supervisados:
1. **Regresión Logística**  
2. **Random Forest**  

Finalmente, se evaluó su desempeño (AUC, precisión, recall, F1‐score) y se midió el grado de acuerdo entre sus predicciones mediante el coeficiente Kappa de Cohen.

---

## Estructura del repositorio
Todos los archivos están ubicados en la raíz del repositorio (no hay carpetas adicionales):

```
ProyectoFinal_Progra/
├─ comparacion_modelos.ipynb
├─ exploracion.ipynb
├─ modelo_1.ipynb
├─ modelo_2.ipynb
├─ preprocesado.ipynb
├─ dataset_preprocesado.csv
└─ README.md 
```

- `exploracion.ipynb`: Notebook con el análisis exploratorio de datos.
- `preprocesado.ipynb`: Notebook para simular e imputar valores faltantes, balancear clases y escalar la variable `Amount`.
- `modelo_1.ipynb`: Notebook con Regresión Logística, ajuste de hiperparámetros y evaluación.
- `modelo_2.ipynb`: Notebook con Random Forest, ajuste de hiperparámetros y evaluación.
- `comparacion_modelos.ipynb`: Notebook que compara los resultados de ambos modelos y calcula el coeficiente Kappa de Cohen.
- `dataset_preprocesado.csv`: Archivo CSV generado tras el preprocesado (balanceo, escalado e imputación).
- `README.md`: Documentación de este repositorio.

---

## Dataset original (no incluido)
El archivo `creditcard.csv` original no está en este repositorio porque supera el límite de tamaño.  
Puedes descargarlo directamente desde Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Una vez lo descargues, debes **subirlo manualmente** a Google Colab (o guardarlo temporalmente en la sesión) siguiendo las instrucciones de cada notebook.

---

## Uso y ejecución en Google Colab
Este proyecto se desarrolló y probó en **Google Colab**.  
Al ejecutar cada notebook en Colab, ten en cuenta:

1. **Subir los datasets necesarios**  
   - Si el notebook requiere el **dataset original**, súbelo con la opción de carga de archivos de Colab.
   - Si el notebook requiere el **dataset preprocesado** (`dataset_preprocesado.csv`), súbelo nuevamente en cada sesión porque la memoria de Colab es temporal y se borra al cerrar la sesión.

2. **Orden de ejecución de los notebooks**  
   1. `exploracion.ipynb`  
   2. `preprocesado.ipynb`  
   3. `modelo_1.ipynb`  
   4. `modelo_2.ipynb`  
   5. `comparacion_modelos.ipynb`  

   Ejecuta cada celda secuencialmente. En la primera celda de cada notebook encontrarás los `import` necesarios.

3. **Rutas de archivos en Colab**  
   - Al usar `pd.read_csv()`, Colab asumirá que el archivo está en la ruta `/content/`.  
   - Por ejemplo:
     ```python
     df = pd.read_csv('creditcard.csv')
     ```
     si subiste `creditcard.csv` a Colab (quedará en `/content/creditcard.csv`).  
   - De manera similar, para el dataset preprocesado:
     ```python
     df = pd.read_csv('dataset_preprocesado.csv')
     ```

4. **Memoria temporal de Colab**  
   - Cada vez que inicies una nueva sesión en Colab, debes volver a subir todos los archivos (dataset original y/o preprocesado).  
   - Verifica con `!ls` o simplemente intenta leer el CSV para confirmar que Colab lo haya recibido correctamente.

---

## Notebooks en detalle

### 1. exploracion.ipynb
- Carga el CSV original (`creditcard.csv`).
- Muestra la forma del DataFrame, información general (`df.info()`), estadísticas descriptivas (`df.describe()`).
- Visualizaciones:
  - Distribución de la variable `Amount`.
  - Distribución de la clase objetivo (`Class`).
  - Mapa de calor de correlaciones entre variables.
  - Detección de outliers con boxplot de `Amount`.
  - Análisis de transacciones por hora (convertir columna `Time` a hora del día si está presente).

### 2. preprocesado.ipynb
- Carga el CSV original (`creditcard.csv`).
- Simula el 5 % de valores faltantes aleatoriamente y los imputa con la mediana (`SimpleImputer(strategy='median')`).
- Balancea las clases mediante submuestreo (la cantidad de fraudes = cantidad de no fraudes).
- Escala la variable `Amount` con `StandardScaler()`.
- Guarda el resultado en `dataset_preprocesado.csv`.

### 3. modelo_1.ipynb
- Carga `dataset_preprocesado.csv`.
- Divide en `X` (características) e `y` (etiqueta `Class`).
- `train_test_split` estratificado (80 % entrenamiento, 20 % prueba).
- Ajusta un modelo de **Regresión Logística** con `GridSearchCV` para optimizar `C` y `penalty`.
- Evalúa el mejor modelo en prueba: métricas AUC, precisión, recall, F1‐score; muestra la matriz de confusión y la curva ROC.
- Grafica la curva de aprendizaje (AUC de entrenamiento vs. validación).

### 4. modelo_2.ipynb
- Carga `dataset_preprocesado.csv`.
- Divide en entrenamiento/prueba de forma estratificada.
- Ajusta un modelo de **Random Forest** con `GridSearchCV` para optimizar `n_estimators`, `max_depth`, `min_samples_split` y `min_samples_leaf`.
- Evalúa en prueba: métricas AUC, precisión, recall, F1‐score; muestra la matriz de confusión y la curva ROC.
- Grafica la curva de aprendizaje.

### 5. comparacion_modelos.ipynb
- Carga `dataset_preprocesado.csv` y divide en entrenamiento/prueba.
- Repite (o reutiliza) `GridSearchCV` para ambos modelos y entrena con los mejores parámetros.
- Calcula métricas comparativas y las organiza en una tabla:
  - AUC, precisión, recall, F1‐score para Regresión Logística y Random Forest.
- Calcula el **coeficiente Kappa de Cohen** entre ambas predicciones binarias.
- Incluye conclusiones sobre cuál modelo conviene usar según las métricas y la interpretabilidad.

---

## Descarga del dataset original
El archivo `creditcard.csv` debe descargarse manualmente desde Kaggle, ya que no se incluye en este repositorio:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

¡Gracias por usar este proyecto! Si tienes dudas adicionales, revisa la documentación dentro de cada notebook o abre un “issue” en el repositorio.
