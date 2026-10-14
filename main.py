"""50 Startups: fit every feature subset and rank them by test MSE.

Run:
    python main.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlr_workflow import (
    RANDOM_STATE,
    TEST_SIZE,
    encode,
    evaluate_subset,
    load_frame,
)

ROOT = Path(__file__).resolve().parent
DATA_URL = (
    "https://raw.githubusercontent.com/Avik-Jain/"
    "100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
)
TARGET = "Profit"


def public_scheme(rank: int, model: dict, mse: float) -> dict:
    return {
        "rank": rank,
        "k": model["k"],
        "features": model["features"],
        "mse": mse,
        "rmse": model["rmse"],
        "mae": model["mae"],
        "r2": model["r2"],
        "adjR2": model["adjR2"],
        "fStat": model["fStat"],
        "fP": model["fP"],
        "coefficients": model["coefficients"],
        "residuals": model["residuals"],
    }


def main():
    frame = load_frame(DATA_URL)
    X, y, _schema, _reference = encode(frame, TARGET)
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("50 Startups multiple linear regression")
    print(f"Train {len(X_train)} / Test {len(X_test)}  |  random_state={RANDOM_STATE}")
    print(f"Candidate features ({len(features)}): {features}")

    ranked = []
    for k in range(1, len(features) + 1):
        for combo in itertools.combinations(features, k):
            chosen = list(combo)
            model = evaluate_subset(chosen, X_train, X_test, y_train, y_test)
            mse = float(mean_squared_error(y_test, model["testPredicted"]))
            ranked.append((mse, model))
    ranked.sort(key=lambda item: (item[0], item[1]["k"]))

    full = next(model for mse, model in ranked if model["k"] == len(features))
    full_mse = next(mse for mse, model in ranked if model["k"] == len(features))
    best_mse, best = ranked[0]

    print("\nTest MSE")
    print(f"Full model ({len(features)} features): {full_mse:.6f}")
    print(f"Best scheme: {best_mse:.6f}")
    print(f"Best features: {best['features']}")

    print("\nTop 20 feature selection schemes (lower test MSE is better)")
    print(f"{'Rank':>4}  {'k':>2}  {'Test MSE':>16}  {'Test RMSE':>12}  {'R2':>8}  Features")
    top20_rows = []
    for rank, (mse, model) in enumerate(ranked[:20], start=1):
        print(
            f"{rank:4d}  {model['k']:2d}  {mse:16.4f}  {model['rmse']:12.4f}  "
            f"{model['r2']:8.4f}  {model['features']}"
        )
        top20_rows.append(
            {
                "rank": rank,
                "k": model["k"],
                "features": model["features"],
                "mse": mse,
                "rmse": model["rmse"],
                "r2": model["r2"],
            }
        )

    best5 = [public_scheme(rank, model, mse) for rank, (mse, model) in enumerate(ranked[:5], start=1)]
    print("\nBest 5 schemes")
    for scheme in best5:
        print(f"  {scheme['rank']}. k={scheme['k']}  MSE={scheme['mse']:.4f}  {scheme['features']}")

    preset_path = ROOT / "data" / "presets.json"
    presets = json.loads(preset_path.read_text(encoding="utf-8"))
    startups = next(item for item in presets if item["id"] == "startups")
    startups["testMse"] = full_mse
    startups["bestTestMse"] = best_mse
    startups["top20"] = top20_rows
    startups["bestSchemes"] = best5
    startups["deployed"]["algorithm"] = "Best scheme 1"
    startups["deployed"]["k"] = best["k"]
    startups["deployed"]["features"] = best["features"]
    startups["deployed"]["rmse"] = best["rmse"]
    startups["deployed"]["r2"] = best["r2"]
    startups["deployed"]["coefficients"] = best["coefficients"]
    preset_path.write_text(json.dumps(presets, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote best 5 schemes into {preset_path}")
    print(f"Full-model test MSE kept for reference: {full['rmse'] ** 2:.6f}")


if __name__ == "__main__":
    main()
