# Freight Rate Prediction

A reproducible solution for the Spotter Machine Learning Engineer assessment. This repository includes the final predictions, the required December chart, and the source code used to generate them. No Jupyter notebook is required.

## Contents

| File | Purpose |
| --- | --- |
| `model_pipeline.py` | Data checks, feature engineering, model selection, time-based holdout evaluation, and final inference |
| `validation_predictions.csv` | Predictions for all 12,000 loads, with exactly `load_id,predicted_rate` |
| `december_predictions.csv` | Daily December 2025 predictions in the original seven-column format |
| `scorer_results/candidate_december.png` | Fixed December chart created by the supplied `score.py` |
| `Spotter_Freight_Rate_Assessment_Report.pdf` | Technical report with measured metrics, data-quality findings, validation design, and the required chart |
| `results/evaluation.json` | Measured data audit, September model comparisons, and October holdout results |
| `requirements.txt` | Python packages |
| `score.py` | Original scorer supplied with the assessment |

## Reproduce from the original assessment data

Use Python 3.11–3.13. Place the following unmodified employer-provided files in a folder named `data` at the repository root:

```text
data/train-test.csv
data/validation.csv
data/validation-predictions-template.csv
data/december-chart-inputs.csv
```

Install dependencies, run the pipeline, then validate output files:

```bash
python -m pip install -r requirements.txt
python model_pipeline.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

Expected scorer confirmation:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

The script also writes `results/evaluation.json` and `results/october_holdout_predictions.csv` for offline evaluation. The latter is a local generated artifact and is not needed for submission.

## Modeling and verification

- The development period is January–October 2025; predictions cover November–December 2025.
- January–August is used to train four LightGBM candidates. September controls early stopping and model selection by dollar MAE. October is held out from those decisions.
- The selected model is **reduced_log**, using pickup, delivery, equipment, mileage, weight, and calendar-derived features. This log-target estimator is refitted on all 48,000 labeled development rows before producing final predictions.
- **October holdout: MAE $115.66, RMSE $650.54, WAPE 4.86%, R² 0.819.** These are computed from the supplied labeled development data, not the hidden 12,000-row evaluation set.
- The December fixed-lane forecast uses the same selected model. Its provided inputs contain no live market index or quote signal, and none are fabricated.
- Nonpositive weights are treated as missing. Unknown categorical values are supported by LightGBM's training-defined category mapping.

The included `score.py` validates output schema, IDs and fixed December inputs, then renders the chart. It does **not** calculate hidden final prediction accuracy.

## Submission notes

The original assessment CSVs are intentionally not republished. Reviewers can reproduce the result using their own copies of the same files. The assessment also requests a separate 2–3 minute camera-on Loom walkthrough submitted in the application form.
