"""Reglas de decisión del protocolo (corrección del 14/11/2026): Gini Stability fuera de partición.

El Gini Stability de cada partición (10-14 semanas) lo domina la penalización por pendiente, así que
se decide con la serie continua de las semanas 40-73 (P1, P2 y P3 de validación unidas, cada semana
predicha por un modelo que no la usó en su ajuste), más el Gini semanal medio por partición.
"""

from __future__ import annotations

import hashlib

import polars as pl

MIN_FOLDS = 2
RULE_TEXT = (
    "oof-gs-v2: select = simplest config unless challenger has higher OOF Gini Stability (weeks 40-73) "
    "AND higher per-fold mean weekly Gini in >=2 of 3 folds; "
    "ablation retire = OOF GS not worse AND per-fold mean weekly Gini not worse in >=2 of 3 folds "
    "AND OOF weekly Gini std not higher"
)
RULE_HASH = hashlib.sha256(RULE_TEXT.encode()).hexdigest()


def _by_fold(folds: pl.DataFrame, config_id: str) -> list[float]:
    return folds.filter(pl.col("config_id") == config_id).sort("fold")["mean_weekly_gini"].to_list()


def select_config(oof: pl.DataFrame, folds: pl.DataFrame, model: str, return_trace: bool = False):
    """Parte de la config más simple; una más compleja la reemplaza solo si su GS fuera de partición es
    mayor y su Gini semanal medio es mayor en al menos 2 de 3 particiones."""
    t = oof.filter(pl.col("model") == model).sort("complexity")
    gs = dict(zip(t["config_id"].to_list(), t["oof_gini_stability"].to_list()))
    ids = t["config_id"].to_list()
    best, trace = ids[0], []
    for c in ids[1:]:
        wins = sum(a > b for a, b in zip(_by_fold(folds, c), _by_fold(folds, best)))
        replaced = bool(gs[c] > gs[best] and wins >= MIN_FOLDS)
        trace.append({"challenger": c, "incumbent": best, "oof_gs_challenger": gs[c], "oof_gs_incumbent": gs[best],
                      "folds_won": int(wins), "replaced": replaced})
        if replaced:
            best = c
    return (best, trace) if return_trace else best


def ablation_decision(base_oof: dict, base_folds: pl.DataFrame, without_oof: dict, without_folds: pl.DataFrame) -> dict:
    """Retirar el grupo si sin él: GS fuera de partición no peor, Gini semanal medio no peor en >= 2 de 3
    particiones y desviación del Gini semanal fuera de partición no mayor."""
    b, w = base_folds.sort("fold")["mean_weekly_gini"].to_list(), without_folds.sort("fold")["mean_weekly_gini"].to_list()
    not_worse = sum(x >= y for x, y in zip(w, b))
    gs_ok = without_oof["oof_gini_stability"] >= base_oof["oof_gini_stability"]
    std_ok = without_oof["std_weekly_gini"] <= base_oof["std_weekly_gini"]
    return {
        "oof_gs_base": float(base_oof["oof_gini_stability"]),
        "oof_gs_without": float(without_oof["oof_gini_stability"]),
        "oof_gs_not_worse": bool(gs_ok),
        "folds_mwg_not_worse": int(not_worse),
        "oof_std_base": float(base_oof["std_weekly_gini"]),
        "oof_std_without": float(without_oof["std_weekly_gini"]),
        "std_not_higher": bool(std_ok),
        "retire": bool(gs_ok and not_worse >= MIN_FOLDS and std_ok),
    }
