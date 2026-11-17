# Brief for the slide deck: Chapter 4 – Modeling (Home Credit, CRISP-DM)

## Goal
Build a slide deck (PPTX + PDF export) into `presentacion/codex/` ONLY (do not touch any other path) for the "Capítulo 4: Modelamiento" progress report of the
DataMine credit-risk project. Language of the slides: neutral professional Spanish (no voseo, no slang).
Present the modeling as completed work; never state or imply it was not executed.

## Reference
`Segundo Avance.pdf` (repo root) is the STRUCTURAL reference only (another group's H&M project).
Follow its sequence and density of slides, but CHANGE colors, typography and layout style.
Do not copy its text, figures or numbers.

## Slide sequence (mirror the reference)
1. Cover: Universidad Nacional Mayor de San Marcos, FISI; project title on credit risk stability;
   "Segundo avance del proyecto integrador · Capítulo 4: Modelamiento"; CRISP-DM fase 4;
   teacher Pedro Martin Lezama Gonzales; members: Nick Salcedo, Gabriel Davalos, Andre Cuenca,
   Giancarlo Villavicencio, Mathias Torres; Lima, Perú 2026. Do not invent a group number.
2. Methodological placement: six CRISP-DM phases, phase 4 highlighted; 4.1 Selección · 4.2 Plan de
   pruebas · 4.3 Construcción · 4.4 Evaluación.
3. Objectives (from `development/01-entendimiento-empresarial.md`, sections 1.1 and 1.3):
   general objective, ON1–ON3 with success criteria, OM1–OM4 with "Sirve a" and criteria.
4. Selection of techniques per objective (table): OM1 logistic regression (base) → Random Forest →
   LightGBM; OM2 SHAP-type contributions; OM3 threshold and calibration curves; OM4 PSI + ADWIN.
5. (removed: no bibliography slide)
6. Test plan: temporal split (fit WEEK_NUM 0–63, selection 64–73, refit 0–73, OOT 74–91 = 18 weeks),
   case counts and positive rates (train 1 332 001 cases, 3.28 %; OOT 194 658, 2.19 %).
7. Metrics and comparability: Gini Stability (formula), AUC/Gini, PR-AUC, FPR at recall 0.80,
   Brier and reliability curve, PSI thresholds 0.10/0.25, ADWIN delta 0.002.
8. Input variables: 94 predictors from previous applications and person tables plus coverage
   indicators; linear view 234 columns (imputed, scaled, one-hot), tree view 114 columns.
9. Configurations and refit: see `experiment_log.jsonl`.
10. Validation comparison (weeks 64–73): `model_comparison.csv` (val_auc).
11. OOT results: `model_comparison.csv`, figure `04_roc_oot.png`.
12. Weekly stability: `weekly_gini.csv`, figure `04_weekly_gini.png`, Gini Stability.
13. Threshold and calibration (OM3): `threshold_sweep.csv`, `calibration_*.csv`,
    figure `04_calibration_oot.png`; limited-history segment (thin_* columns).
14. Explanation (OM2): `shap_importance.csv`, `shap_example_case.csv`, figure `04_shap_top.png`.
    State that contribution does not imply causality.
15. Retraining windows (OM4): `drift_weekly.csv`, `adwin_detections.csv`, figure `04_drift_psi_adwin.png`.
16. Verifications, limitations and decision (selected model and next phase).
17. Thank-you slide. NO references slide and no citations anywhere in the deck.

## Data sources
All numbers and figures come from
`presentacion/insumos/` in the repo (CSV files, `experiment_log.jsonl`, `figures/*.png`, `run_models.py`).
Do not invent numbers; if a value is missing, leave a visible placeholder and list it.

## Key results to highlight (verify against the CSVs)
- Selected model: LightGBM, 63 leaves, 173 trees (lr 0.05, early stopping on weeks 64–73).
- Gini Stability OOT: LightGBM 0.493 vs logistic 0.452 vs random forest 0.377 (OM1 criterion met).
- AUC OOT: 0.757 / 0.738 / 0.733. FPR at recall 0.80: 43.6 % vs 46.4 % (logistic): 5 331 fewer false positives.
- Limited-history segment (no previous applications, 28 758 cases): AUC 0.756, FPR 44.3 % vs 45.8 %.
- Explanation: top-3 factors stable (income type, age, mean tenor of previous applications);
  `tax_registry_b_1__n_records` jumps from rank 59 (train) to 4 (OOT): coverage change.
- Retraining windows: score PSI 0.025–0.060 (below 0.10); ADWIN on score flags weeks 82, 85, 89;
  ADWIN on per-case loss flags weeks 87, 90; weakest weekly Gini in weeks 81–82 and 84.
- The MLP was not trained: mention only as future work, never as a result.
- Figure `04_drift_psi_adwin.png` lacks a legend for the ADWIN lines (purple = score, green = loss):
  explain it in the slide text.
