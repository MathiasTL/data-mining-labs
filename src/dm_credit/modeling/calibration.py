"""Calibración isotónica ajustada con las predicciones de P3 (semanas 60-73).

Se elige isotónica sobre Platt: con ~113k casos y ~4 mil positivos en la validación hay datos de
sobra para una función monótona no paramétrica, y LightGBM/LR sesgan la probabilidad de forma
no necesariamente sigmoide. Se recorta fuera de rango (sin extrapolar). El Gini y los umbrales
usan el score crudo porque la isotónica crea empates; la calibración usa el score calibrado.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6


def fit_calibrator(y, p) -> IsotonicRegression:
    cal = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip", increasing=True)
    cal.fit(np.asarray(p, float), np.asarray(y, float))
    return cal
