from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CATEGORICAL = ("pickup", "delivery", "equipment")
CORE = ("pickup", "delivery", "distance", "equipment", "weight", "date")
EXTRA = ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal")


def load_data():
    names = {
        "train": "train-test.csv",
        "validation": "validation.csv",
        "template": "validation-predictions-template.csv",
        "december": "december-chart-inputs.csv",
    }
    frames = {key: pd.read_csv(DATA / filename) for key, filename in names.items()}
    train, validation, template, december = (frames[key] for key in names)
    expected = {
        "train": set(CORE) | set(EXTRA) | {"load_id", "posted_rate"},
        "validation": set(CORE) | set(EXTRA) | {"load_id"},
        "template": {"load_id", "predicted_rate"},
        "december": set(CORE) | {"predicted_rate"},
    }
    for name, frame in frames.items():
        missing = expected[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if (len(train), len(validation), len(template), len(december)) != (48000, 12000, 12000, 31):
        raise ValueError("Unexpected dataset sizes")
    for name in ("train", "validation", "template"):
        frame = frames[name]
        if frame.load_id.isna().any() or frame.load_id.duplicated().any():
            raise ValueError(f"Invalid load_id in {name}")
    if set(template.load_id) != set(validation.load_id):
        raise ValueError("Template load IDs differ from validation load IDs")
    rates = pd.to_numeric(train.posted_rate, errors="coerce")
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Training rates must be finite and positive")
    for name in ("train", "validation", "december"):
        frame = frames[name]
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        miles = pd.to_numeric(frame.distance, errors="coerce")
        if not np.isfinite(miles).all() or (miles <= 0).any():
            raise ValueError(f"Invalid distance in {name}")
    if train.date.max() >= validation.date.min():
        raise ValueError("Prediction window overlaps training dates")
    train = train.sort_values(["date", "load_id"]).reset_index(drop=True)
    return train, validation, template, december


def make_features(data, use_extra):
    x = data[list(CATEGORICAL)].fillna("Unknown").astype(str).copy()
    miles = pd.to_numeric(data.distance, errors="coerce")
    weight = pd.to_numeric(data.weight, errors="coerce").mask(lambda values: values <= 0)
    date = data.date
    x["distance"] = miles
    x["weight"] = weight
    x["log_distance"] = np.log1p(miles)
    x["weight_per_mile"] = (weight / miles).clip(0, 2000)
    x["elapsed_days"] = (date - pd.Timestamp("2025-01-01")).dt.days
    x["month"] = date.dt.month
    x["weekday"] = date.dt.weekday
    x["day_of_month"] = date.dt.day
    x["day_of_year"] = date.dt.dayofyear
    x["week_of_year"] = date.dt.isocalendar().week.astype(int)
    x["weekday_sin"] = np.sin(2 * np.pi * date.dt.weekday / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * date.dt.weekday / 7)
    x["annual_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    x["annual_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)
    if use_extra:
        for column in EXTRA:
            x[column] = pd.to_numeric(data[column], errors="coerce")
        x["latitude_delta"] = x.delivery_lat - x.pickup_lat
        x["longitude_delta"] = x.delivery_lon - x.pickup_lon
        lat1 = np.radians(x.pickup_lat)
        lat2 = np.radians(x.delivery_lat)
        haversine = np.sin((lat2 - lat1) / 2) ** 2 + (
            np.cos(lat1) * np.cos(lat2) * np.sin(np.radians(x.longitude_delta) / 2) ** 2
        )
        x["great_circle_miles"] = 7917.6 * np.arcsin(np.sqrt(haversine.clip(0, 1)))
        x["road_to_geo_ratio"] = (miles / x.great_circle_miles.clip(lower=25)).clip(0, 40)
        x["distance_x_market"] = miles * x.market_index
        x["distance_x_quote"] = miles * x.quote_signal
        x["market_x_quote"] = x.market_index * x.quote_signal
    return x


def with_categories(x, levels):
    result = x.copy()
    for name in CATEGORICAL:
        result[name] = pd.Categorical(result[name], categories=levels[name])
    return result


def train_model(kind, x, y, trees, validation=None):
    levels = {name: sorted(x[name].unique().tolist()) for name in CATEGORICAL}
    regressor = LGBMRegressor(
        objective="regression",
        n_estimators=trees,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=65,
        colsample_bytree=0.92,
        reg_lambda=6.0,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )
    target = np.log1p(y) if kind == "log" else y
    fit_options = {"categorical_feature": list(CATEGORICAL)}
    if validation is not None:
        x_val, y_val = validation
        val_target = np.log1p(y_val) if kind == "log" else y_val
        fit_options["eval_set"] = [(with_categories(x_val, levels), val_target)]
        fit_options["eval_metric"] = "rmse"
        fit_options["callbacks"] = [early_stopping(85, first_metric_only=True, verbose=False)]
    regressor.fit(with_categories(x, levels), target, **fit_options)
    return regressor, levels


def predict(kind, regressor, levels, x):
    values = np.asarray(regressor.predict(with_categories(x, levels)), dtype=float)
    if kind == "log":
        values = np.expm1(values)
    return np.maximum(values, 1.0)


def metrics(actual, estimate):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(estimate, dtype=float)
    return {
        "mae": float(mean_absolute_error(a, p)),
        "rmse": float(np.sqrt(mean_squared_error(a, p))),
        "r2": float(r2_score(a, p)),
        "wape_percent": float(100 * np.abs(a - p).sum() / np.abs(a).sum()),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    train, validation, template, december = load_data()
    train_september = train.date < "2025-09-01"
    september = (train.date >= "2025-09-01") & (train.date < "2025-10-01")
    train_october = train.date < "2025-10-01"
    october = (train.date >= "2025-10-01") & (train.date < "2025-11-01")
    if not all(mask.any() for mask in (train_september, september, train_october, october)):
        raise ValueError("Missing chronological evaluation period")
    y = train.posted_rate
    feature_sets = {"reduced": make_features(train, False), "full": make_features(train, True)}
    candidates = {}
    for feature_name, x in feature_sets.items():
        for target_kind in ("raw", "log"):
            model, levels = train_model(
                target_kind, x.loc[train_september], y.loc[train_september], 900,
                (x.loc[september], y.loc[september]),
            )
            count = int(model.best_iteration_)
            if not 1 <= count <= 900:
                raise ValueError("Invalid early stopping result")
            forecast = predict(target_kind, model, levels, x.loc[september])
            candidates[f"{feature_name}_{target_kind}"] = {
                "trees": count, **metrics(y.loc[september], forecast)
            }
    selected = min(candidates, key=lambda name: candidates[name]["mae"])
    feature_name, target_kind = selected.split("_")
    x = feature_sets[feature_name]
    trees = candidates[selected]["trees"]
    model_oct, levels_oct = train_model(target_kind, x.loc[train_october], y.loc[train_october], trees)
    oct_values = predict(target_kind, model_oct, levels_oct, x.loc[october])
    oct_metrics = metrics(y.loc[october], oct_values)
    baseline_per_mile = float(np.median(y.loc[train_october] / train.loc[train_october, "distance"]))
    baseline_oct = metrics(y.loc[october], baseline_per_mile * train.loc[october, "distance"])
    final_model, levels = train_model(target_kind, x, y, trees)
    final_values = predict(target_kind, final_model, levels, make_features(validation, feature_name == "full"))
    forecasts = validation[["load_id"]].copy()
    forecasts["predicted_rate"] = np.round(final_values, 2)
    output = template[["load_id"]].merge(forecasts, on="load_id", how="left", validate="one_to_one")
    if output.predicted_rate.isna().any() or not np.isfinite(output.predicted_rate).all() or (output.predicted_rate <= 0).any():
        raise ValueError("Final predictions are invalid")
    output.to_csv(ROOT / "validation_predictions.csv", index=False)
    if feature_name == "reduced":
        chart_model, chart_levels = final_model, levels
        chart_kind = target_kind
        chart_model_details = "same final reduced-feature model"
    else:
        chart_model, chart_levels = train_model("log", feature_sets["reduced"], y, 120)
        chart_kind = "log"
        chart_model_details = "separate reduced-feature LightGBM model (120 trees)"
    december_values = predict(chart_kind, chart_model, chart_levels, make_features(december, False))
    if not np.isfinite(december_values).all() or (december_values <= 0).any():
        raise ValueError("December predictions are invalid")
    december["predicted_rate"] = np.round(december_values, 2)
    december[list(CORE) + ["predicted_rate"]].assign(
        date=december.date.dt.strftime("%Y-%m-%d")
    ).to_csv(ROOT / "december_predictions.csv", index=False)
    pd.DataFrame({
        "date": train.loc[october, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_rate": y.loc[october].to_numpy(),
        "predicted_rate": oct_values,
    }).to_csv(RESULTS / "october_holdout_predictions.csv", index=False)
    audit = {
        "training_rows": int(len(train)),
        "prediction_rows": int(len(validation)),
        "december_rows": int(len(december)),
        "training_date_range": [str(train.date.min().date()), str(train.date.max().date())],
        "prediction_date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
        "train_months": {str(key): int(val) for key, val in train.date.dt.to_period("M").value_counts().sort_index().items()},
        "missing_weight_train": int(train.weight.isna().sum()),
        "missing_weight_validation": int(validation.weight.isna().sum()),
        "nonpositive_weight_train": int((train.weight <= 0).sum()),
        "nonpositive_weight_validation": int((validation.weight <= 0).sum()),
        "missing_market_index_train": int(train.market_index.isna().sum()),
        "missing_market_index_validation": int(validation.market_index.isna().sum()),
        "unseen_cities": sorted((set(validation.pickup.dropna()) | set(validation.delivery.dropna())) - (set(train.pickup.dropna()) | set(train.delivery.dropna()))),
        "min_training_rate": float(y.min()),
        "max_training_rate": float(y.max()),
        "training_rate_over_10000": int((y > 10000).sum()),
        "selection_training_rows": int(train_september.sum()),
        "september_validation_rows": int(september.sum()),
        "october_holdout_rows": int(october.sum()),
    }
    evaluation = {
        "model_selection_metric": "September MAE in USD",
        "candidates": candidates,
        "selected_model": selected,
        "october_holdout": oct_metrics,
        "october_baseline": baseline_oct,
        "baseline_dollars_per_mile": baseline_per_mile,
        "december_model": chart_model_details,
        "feature_importance_gain": dict(sorted(zip(x.columns, map(float, final_model.booster_.feature_importance(importance_type="gain"))), key=lambda item: item[1], reverse=True)),
        "validation_prediction_min": float(final_values.min()),
        "validation_prediction_max": float(final_values.max()),
        "december_prediction_min": float(december_values.min()),
        "december_prediction_max": float(december_values.max()),
    }
    (RESULTS / "evaluation.json").write_text(json.dumps({"audit": audit, "evaluation": evaluation}, indent=2), encoding="utf-8")
    print(f"Selected: {selected} ({trees} trees)")
    print(f"October holdout: MAE ${oct_metrics['mae']:.2f}, RMSE ${oct_metrics['rmse']:.2f}, WAPE {oct_metrics['wape_percent']:.2f}%")
    print(f"Wrote {len(output):,} validation predictions and {len(december):,} December predictions")


if __name__ == "__main__":
    main()
