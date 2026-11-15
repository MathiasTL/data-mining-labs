"""Modelos del protocolo: regresión logística (referencia) y LightGBM (candidato).

Rejilla pequeña y comparable (4 vs 2 configuraciones se mantienen simples). Sin early stopping:
``n_estimators`` es parte de la rejilla. LightGBM entrena el máximo de árboles una vez y evalúa
prefijos (``num_iteration``), equivalente a entrenar cada tamaño por separado con la misma semilla.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
N_JOBS = 8
LR_MAX_ITER = 300


def set_n_jobs(n: int) -> None:
    global N_JOBS
    if n < 1:
        raise ValueError("n_jobs must be >= 1")
    N_JOBS = int(n)
LR_GRID_C = (0.1, 1.0)
LGBM_LEAVES = (31, 63)
LGBM_TREES = (300, 600)
LGBM_FIXED = {
    "objective": "binary",
    "learning_rate": 0.05,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_bin": 255,
}
CONSTANT = "constant"
VIEW_OF = {"logistic": "linear", "lightgbm": "tree"}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    params: dict
    complexity: int = 0  # menor = más simple (desempate del protocolo)

    @property
    def config_id(self) -> str:
        return self.model + ":" + json.dumps(self.params, sort_keys=True, separators=(",", ":"))


def constant_config() -> ModelConfig:
    """Referencia constante: tasa base del bloque de ajuste. No es candidato de selección."""
    return ModelConfig(CONSTANT, {}, 0)


def grid(model: str) -> list[ModelConfig]:
    if model == "logistic":
        # C pequeño = más regularizado = más simple
        return [ModelConfig("logistic", {"C": c}, i) for i, c in enumerate(LR_GRID_C)]
    if model == "lightgbm":
        out, i = [], 0
        for leaves in LGBM_LEAVES:
            for trees in LGBM_TREES:
                out.append(ModelConfig("lightgbm", {**LGBM_FIXED, "num_leaves": leaves, "n_estimators": trees}, i))
                i += 1
        return out
    raise ValueError(f"unknown model {model!r}")


@dataclass
class FittedModel:
    kind: str
    obj: object
    names: list
    cat_idx: list
    view: str
    info: dict | None = None  # p. ej. n_iter y convergencia de la logística

    def predict(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        if self.kind == "logistic":
            return self.obj.predict_proba(X)[:, 1]
        return self.obj.predict(X, num_iteration=n_estimators)


def fit_model(cfg: ModelConfig, ctx, *, n_estimators: int | None = None) -> FittedModel:
    if cfg.model == CONSTANT:
        return FittedModel(CONSTANT, float(ctx.y_fit.mean()), [], [], None, {"fit_seconds": 0.0, "predict_seconds": 0.0})
    view = VIEW_OF[cfg.model]
    X, names, cat_idx = ctx.matrix(view, "fit")
    y = ctx.y_fit
    t0 = time.perf_counter()
    if cfg.model == "logistic":
        max_iter = cfg.params.get("max_iter", LR_MAX_ITER)
        clf = LogisticRegression(C=cfg.params["C"], solver="lbfgs", max_iter=max_iter)
        clf.fit(X, y)
        n_iter = int(np.max(clf.n_iter_))
        return FittedModel("logistic", clf, names, cat_idx, view,
                           {"n_iter": n_iter, "converged": bool(n_iter < max_iter),
                            "fit_seconds": time.perf_counter() - t0, "predict_seconds": 0.0})
    params = {k: v for k, v in cfg.params.items() if k != "n_estimators"}
    params.update(
        seed=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED,
        deterministic=True, force_row_wise=True, num_threads=N_JOBS, verbose=-1,
    )
    ds = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=cat_idx, free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=n_estimators or cfg.params["n_estimators"])
    return FittedModel("lightgbm", booster, names, cat_idx, view,
                       {"fit_seconds": time.perf_counter() - t0, "predict_seconds": 0.0})


def fit_predict(cfg: ModelConfig, ctx, blocks, *, n_estimators_eval=None, return_model: bool = False):
    """Entrena con el bloque de ajuste y predice ``blocks``. Claves: bloque, o (bloque, n) con prefijos.

    ``info["fit_seconds"]`` mide solo el entrenamiento; ``info["predict_seconds"]`` solo ``predict``
    (sin construir las matrices).
    """
    top = max(n_estimators_eval) if n_estimators_eval else None
    model = fit_model(cfg, ctx, n_estimators=top)
    preds, spent = {}, 0.0
    for block in blocks:
        if model.kind == CONSTANT:
            preds[block] = np.full(len(ctx.y(block)), model.obj)
            continue
        X, _, _ = ctx.matrix(model.view, block)
        t0 = time.perf_counter()
        if n_estimators_eval:
            for n in n_estimators_eval:
                preds[(block, n)] = model.predict(X, n)
        else:
            preds[block] = model.predict(X)
        spent += time.perf_counter() - t0
        del X
    model.info["predict_seconds"] = spent
    return (preds, model) if return_model else preds
