"""Thí nghiệm bổ sung (không ảnh hưởng mô hình chính thức):

1. Độ ổn định theo seed: XGBoost và LightGBM, 5 seed, bộ đặc trưng chính "base".
2. Mã hoá loại ngày lễ: so sánh bộ "base" với "holiday_type" (thêm StateHoliday a/b/c), XGBoost 5 seed.

Kết luận chỉ dựa trên Validation; cột Test để tham khảo.
Chạy: python scripts/experiments.py   (~2–3 phút) -> results/evaluation/experiment_*.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.loader import clean_train, load_store, load_train, split_by_time  # noqa: E402
from src.features.pipeline import RossmannPipeline, inverse_target, target_log  # noqa: E402
from src.models.evaluate import evaluate_predictions  # noqa: E402
from src.models.train import PARAMS, make_model  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]


def run(feature_set, model_names, train, val, test, store):
    pipe = RossmannPipeline(feature_set).fit(train, store)
    Xtr, Xva, Xte = pipe.transform(train), pipe.transform(val), pipe.transform(test)
    rows = []
    for name in model_names:
        for seed in SEEDS:
            model = make_model(name, {**PARAMS[name], "random_state": seed}).fit(Xtr, target_log(train.Sales))
            v = evaluate_predictions(val.Sales, inverse_target(model.predict(Xva)))
            t = evaluate_predictions(test.Sales, inverse_target(model.predict(Xte)))
            rows.append({"Model": name, "FeatureSet": feature_set, "Seed": seed, "Val RMSPE": v["RMSPE"],
                         "Test RMSPE": t["RMSPE"], "Test MAE": t["MAE"], "Test RMSE": t["RMSE"]})
            print(f"{name:9s} {feature_set:12s} seed {seed:2d}  Val {v['RMSPE']:.5f}  Test {t['RMSPE']:.5f}")
    return rows


def summary(df):
    return (df.groupby(["Model", "FeatureSet"])[["Val RMSPE", "Test RMSPE", "Test MAE", "Test RMSE"]]
            .agg(["mean", "std", "min", "max"]).round(5))


if __name__ == "__main__":
    train, val, test = split_by_time(clean_train(load_train()))
    store = load_store()
    seed_df = pd.DataFrame(run("base", ["XGBoost", "LightGBM"], train, val, test, store))
    seed_df.to_csv(config.OUT_EVAL / "experiment_seed_stability.csv", index=False)
    fs_df = pd.concat([seed_df[seed_df.Model == "XGBoost"],
                       pd.DataFrame(run("holiday_type", ["XGBoost"], train, val, test, store))])
    fs_df.to_csv(config.OUT_EVAL / "experiment_feature_sets.csv", index=False)
    print("\n== Độ ổn định theo seed ==\n", summary(seed_df).to_string())
    print("\n== Bộ đặc trưng ==\n", summary(fs_df).to_string())
