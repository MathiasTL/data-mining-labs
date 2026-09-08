"""Acumulador de resultados numéricos del pipeline.

Los scripts de análisis escriben aquí toda cifra que después aparecerá en el
informe. El generador del DOCX lee este archivo, de modo que el texto nunca
contradice a los cálculos.
"""
import json
from pathlib import Path

from config import RESULTS_JSON


def _leer_crudo() -> dict:
    if RESULTS_JSON.exists():
        try:
            return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def guardar(seccion: str, clave: str, valor) -> None:
    """Escribe results[seccion][clave] = valor sin borrar el resto del archivo."""
    datos = _leer_crudo()
    datos.setdefault(seccion, {})[clave] = _serializable(valor)
    RESULTS_JSON.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def guardar_varios(seccion: str, pares: dict) -> None:
    for clave, valor in pares.items():
        guardar(seccion, clave, valor)


def cargar(seccion: str | None = None) -> dict:
    datos = _leer_crudo()
    return datos.get(seccion, {}) if seccion else datos


def _serializable(valor):
    """Convierte numpy/pandas a tipos JSON nativos."""
    import numpy as np
    import pandas as pd

    if isinstance(valor, dict):
        return {str(k): _serializable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_serializable(v) for v in valor]
    if isinstance(valor, pd.DataFrame):
        return valor.to_dict(orient="records")
    if isinstance(valor, pd.Series):
        return {str(k): _serializable(v) for k, v in valor.items()}
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return None if np.isnan(valor) else round(float(valor), 4)
    if isinstance(valor, float):
        return None if valor != valor else round(valor, 4)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    return valor


def limpiar_seccion(seccion: str) -> None:
    """Borra una sección completa antes de recalcularla (evita restos viejos)."""
    datos = _leer_crudo()
    datos.pop(seccion, None)
    RESULTS_JSON.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
