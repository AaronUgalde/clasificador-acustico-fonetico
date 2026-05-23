# Clasificador Acústico-Fonético

Práctica 2 de Reconocimiento de Voz. Contiene dos clasificadores basados en árboles de decisión (scikit-learn) para análisis fonético de segmentos de audio.

---

## Clasificadores

### 1. Clasificador de sonidos — `sound_classifier.ipynb`

Clasifica segmentos de audio en tres categorías:

| Clase | Descripción |
|-------|-------------|
| `voice` | Segmento de voz sonora |
| `unvoice` | Segmento de voz sorda (sin vibración glotal) |
| `silence` | Silencio |

**Features usadas:** energía (`energia`) y periodicidad (`periodicidad`).  
**Dataset:** `data/dataset_energia_periodicidad.csv`  
**Modelo entrenado:** `modelos/decision_tree_voice_unvoice_silence.pkl`

El notebook genera las siguientes visualizaciones en `diagramas/`:
- `arbol_decision_sound.png` — estructura del árbol de decisión
- `frontera_decision_sound.png` — frontera de decisión en el espacio energía-periodicidad
- `confusion_matrix_sound.png` — matriz de confusión (conteos y normalizada)
- `metricas_por_clase_sound.png` — precisión, recall y F1 por clase
- `importancia_features_sound.png` — importancia relativa de cada feature (Gini)

---

### 2. Clasificador de vocales — `vocal_classifier.ipynb`

Clasifica las siguientes vocales del inglés (notación ARPABET):

`IY · IH · EH · AE · AH · AA · AO · UH · UW · ER`

**Features usadas:** formantes F1, F2 y F3 (en Hz).  
**Dataset:** `data/dataset_solo_vocales.csv`  
**Modelo entrenado:** `modelos/decision_tree_vocales.pkl`

El notebook genera las siguientes visualizaciones en `diagramas/`:
- `arbol_decision_vocales.png` — estructura del árbol de decisión
- `espacio_vocalico.png` — espacio vocálico F1-F2 (convención fonética estándar)
- `distribucion_vocales.png` — cantidad de muestras por vocal
- `confusion_matrix_vocales.png` — matriz de confusión (conteos y normalizada por vocal)
- `metricas_por_vocal.png` — precisión, recall y F1 por vocal
- `errores_frecuentes_vocales.png` — pares de vocales que el modelo confunde con mayor frecuencia
- `importancia_formantes.png` — importancia relativa de F1, F2 y F3 (Gini)

---

## Estructura del proyecto

```
clasificador-acustico-fonetico/
├── data/
│   ├── dataset_energia_periodicidad.csv   # datos para sound_classifier
│   └── dataset_solo_vocales.csv           # datos para vocal_classifier
├── modelos/
│   ├── decision_tree_voice_unvoice_silence.pkl
│   └── decision_tree_vocales.pkl
├── diagramas/                             # figuras generadas por los notebooks
├── sound_classifier.ipynb
└── vocal_classifier.ipynb
```

---

## Requisitos

```
python >= 3.8
pandas
numpy
matplotlib
scikit-learn
joblib
jupyter
```

Instalación:

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

---

## Ejecución

1. Clona el repositorio y entra al directorio:

```bash
git clone <url-del-repo>
cd clasificador-acustico-fonetico
```

2. Asegúrate de que los datasets estén en `data/`:

```
data/dataset_energia_periodicidad.csv
data/dataset_solo_vocales.csv
```

3. Abre y ejecuta cada notebook en orden (Cell → Run All):

```bash
jupyter notebook sound_classifier.ipynb
jupyter notebook vocal_classifier.ipynb
```

Al terminar la ejecución, los modelos se guardarán automáticamente en `modelos/` y las figuras en `diagramas/`.

---

## Usar los modelos entrenados

Los modelos guardados en `modelos/` pueden cargarse directamente con `joblib` sin necesidad de volver a entrenar.

### Clasificador de sonidos

```python
import joblib
import pandas as pd

modelo = joblib.load("modelos/decision_tree_voice_unvoice_silence.pkl")

# Entrada: [[energia, periodicidad]]
muestra = pd.DataFrame([[0.15, 0.35]], columns=["energia", "periodicidad"])
prediccion = modelo.predict(muestra)
print(prediccion)  # ['voice'] | ['unvoice'] | ['silence']
```

### Clasificador de vocales

```python
import joblib
import pandas as pd

modelo = joblib.load("modelos/decision_tree_vocales.pkl")

# Entrada: [[F1, F2, F3]] en Hz
muestra = pd.DataFrame([[800, 1200, 2500]], columns=["F1", "F2", "F3"])
prediccion = modelo.predict(muestra)
print(prediccion)  # ['AH'] | ['IY'] | etc.
```

> Los nombres de columna deben coincidir con los usados durante el entrenamiento. Si tu CSV usa nombres distintos (por ejemplo `F1_Hz`), pasa los valores como array de NumPy: `modelo.predict([[800, 1200, 2500]])`.
