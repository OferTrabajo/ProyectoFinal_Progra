# Proyecto: Detección de Fraude en Transacciones con Tarjeta de Crédito

## Descripción del problema
Este proyecto tiene como objetivo predecir transacciones fraudulentas en un histórico de tarjetas de crédito utilizando técnicas de Machine Learning. A partir de un conjunto de datos etiquetado con transacciones legítimas y fraudulentas, se implementaron y compararon dos algoritmos supervisados:
1. **Regresión Logística**  
2. **Random Forest**  

Finalmente, se evaluó su desempeño (AUC, precisión, recall, F1‐score) y se midió el grado de acuerdo entre sus predicciones mediante el coeficiente Kappa de Cohen.

## Estructura del repositorio
```
/
├─ notebooks/
│  ├─ 01_exploracion.ipynb
│  ├─ 02_preprocesado.ipynb
│  ├─ 03_modelo_1_regresion_logistica.ipynb
│  ├─ 04_modelo_2_random_forest.ipynb
│  └─ 05_comparacion_modelos.ipynb
├─ data/
│  └─ creditcard.csv
├─ results/
│  └─ dataset_preprocesado.csv
├─ requirements.txt
└─ README.md
```
- `notebooks/`: contiene los Jupyter Notebooks con cada paso del flujo de trabajo.
- `data/creditcard.csv`: el archivo original descargado de Kaggle.
- `results/dataset_preprocesado.csv`: el CSV resultante después de simular e imputar valores, balancear clases y escalar la variable `Amount`.
- `requirements.txt`: listado de dependencias de Python necesarias.
- `README.md`: este archivo de documentación.

---

## Requisitos e instalación

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/tu_usuario/repositorio-fraude-tarjetas.git
   cd repositorio-fraude-tarjetas
   ```

2. **Crear y activar un entorno virtual (opcional pero recomendado)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate       # Windows
   ```

3. **Instalar dependencias**  
   Asegúrate de tener pip actualizado y luego ejecuta:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   El archivo `requirements.txt` contiene, como mínimo:
   ```
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   notebook
   ```
   Si quieres usar JupyterLab, agrega:
   ```
   jupyterlab
   ```

---

## Uso y flujo de trabajo

A continuación se explica el orden en que debes abrir y ejecutar los notebooks. Cada uno depende del resultado del anterior:

1. **01_exploracion.ipynb**  
   - Carga el archivo original `data/creditcard.csv`.  
   - Muestra la forma del DataFrame, estadísticas descriptivas y visualizaciones básicas (histograma de `Amount`, distribución de `Class`, mapa de calor de correlaciones, detección de outliers, análisis por hora).  
   - Objetivo: familiarizarse con los datos, verificar que no hay valores faltantes y entender la naturaleza del problema.

2. **02_preprocesado.ipynb**  
   - Lee el CSV original `data/creditcard.csv`.  
   - Simula el 5 % de valores faltantes (selección aleatoria de celdas) y los imputa con la mediana.  
   - Balancea las clases por submuestreo (iguala la cantidad de fraudes y no fraudes).  
   - Escala la variable `Amount` con `StandardScaler()`.  
   - Genera y guarda `results/dataset_preprocesado.csv`, que luego se usará en los modelos.  
   - Resultado: dataset limpio y balanceado con 924 registros y 32 columnas (incluida la etiqueta).

3. **03_modelo_1_regresion_logistica.ipynb**  
   - Carga `results/dataset_preprocesado.csv`.  
   - Separa en variables `X` (todas las columnas menos `Class`) e `y` (`Class`).  
   - Hace `train_test_split` estratificado (80 % entrenamiento / 20 % prueba).  
   - Ejecuta un `GridSearchCV` para optimizar hiperparámetros de Regresión Logística (`C` y `penalty`).  
   - Entrena el modelo con los mejores parámetros y evalúa en el conjunto de prueba.  
   - Genera métricas (AUC, precisión, recall, F1‐score) y muestra matriz de confusión y curva ROC.  
   - Dibuja la curva de aprendizaje (training vs. validación) para detectar overfitting/underfitting.

4. **04_modelo_2_random_forest.ipynb**  
   - Carga `results/dataset_preprocesado.csv`.  
   - Divide en `X_train`, `X_test`, `y_train`, `y_test` con `train_test_split`.  
   - Ejecuta un `GridSearchCV` para ajustar los hiperparámetros de Random Forest (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).  
   - Entrena el `RandomForestClassifier` con los mejores parámetros y evalúa en test.  
   - Calcula métricas (AUC, precisión, recall, F1‐score) y muestra matriz de confusión y curva ROC.  
   - Grafica la curva de aprendizaje para Random Forest.

5. **05_comparacion_modelos.ipynb**  
   - Vuelve a cargar `dataset_preprocesado.csv` y hace el split estratificado en entrenamiento/prueba.  
   - Repite (o reutiliza) los resultados de GridSearchCV para ambos modelos, entrena cada uno con los mejores parámetros y obtiene las predicciones.  
   - Calcula métricas comparativas y las organiza en un DataFrame:  
     - AUC, precisión, recall y F1‐score de Regresión Logística vs. Random Forest.  
   - Calcula el **coeficiente Kappa de Cohen** entre ambas predicciones para medir concordancia.  
   - Incluye breves conclusiones sobre cuál modelo es más adecuado según métricas y balance interpretabilidad/precisión.

---

## Resumen de notebooks

- **01_exploracion.ipynb**  
  - Análisis exploratorio y entendimiento del dataset original.  
  - Visualizaciones básicas (distribución de montos, proporción de fraudes, correlaciones, outliers, análisis por hora).

- **02_preprocesado.ipynb**  
  - Simulación de valores faltantes (5 %) e imputación con mediana.  
  - Balanceo de clases por submuestreo.  
  - Escalado de la variable `Amount`.  
  - Generación de `dataset_preprocesado.csv`.

- **03_modelo_1_regresion_logistica.ipynb**  
  - GridSearchCV para Regresión Logística.  
  - Evaluación en test: AUC ~ 0.95, precisión ~ 0.95, recall ~ 0.85, F1 ~ 0.90.  
  - Curva ROC y curva de aprendizaje.

- **04_modelo_2_random_forest.ipynb**  
  - GridSearchCV para Random Forest.  
  - Evaluación en test: AUC ~ 0.97, precisión ~ 0.99, recall ~ 0.85, F1 ~ 0.91.  
  - Curva ROC y curva de aprendizaje.

- **05_comparacion_modelos.ipynb**  
  - Comparación directa de métricas en test.  
  - Cálculo de Kappa de Cohen (~ 0.89) para medir concordancia.  
  - Discusión de fortalezas y debilidades de cada enfoque.

---

## Cómo ejecutar cada notebook

1. Asegúrate de estar en el entorno virtual (si lo creaste) y de tener instaladas todas las dependencias.
2. Abre una terminal o consola en la carpeta raíz del proyecto.
3. Inicia Jupyter Notebook o JupyterLab:
   ```bash
   jupyter notebook
   # o, si usas JupyterLab:
   jupyter lab
   ```
4. Navega a `notebooks/` y abre los archivos en el siguiente orden:
   1. `01_exploracion.ipynb`  
   2. `02_preprocesado.ipynb`  
   3. `03_modelo_1_regresion_logistica.ipynb`  
   4. `04_modelo_2_random_forest.ipynb`  
   5. `05_comparacion_modelos.ipynb`

5. Ejecuta cada celda secuencialmente.  
   - En cada notebook, la primera celda importa las librerías necesarias.  
   - Verifica que la ruta de `creditcard.csv` (en `data/`) y la de `dataset_preprocesado.csv` (en `results/`) coincidan con tu estructura de carpetas.  

---

## Dependencias recomendadas

En `requirements.txt` se listan las librerías mínimas requeridas. Un ejemplo de contenido:

```
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
notebook==6.5.2
```


