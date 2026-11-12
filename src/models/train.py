"""ML Layer – huấn luyện, chọn mô hình theo Validation, đánh giá Test, lưu model.

Tương đương file training_model.py trong bài thực hành trên lớp, nhưng tách thành từng bước:
    1. Load dữ liệu           -> loader.load_train / load_store
    2. Làm sạch               -> clean_train (Open != 0 và Sales > 0)
    3. Chia theo thời gian    -> split_by_time (Train / Validation / Test)
    4. Feature Engineering    -> RossmannPipeline.fit(Train)
    5. Baseline               -> Baseline A (TB Store–Thứ–Promo), Baseline B (52 tuần trước)
    6. Train                  -> Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM
                                 (target log1p(Sales)); Prophet chạy riêng: scripts/prophet_experiment.py
    7. Chọn mô hình           -> RMSPE Validation thấp nhất (KHÔNG nhìn Test)
    8. Đánh giá Test          -> tính sau khi đã chọn, chỉ để báo cáo
    9. Lưu                    -> trained/sales_model.joblib + trained/model_metadata.json
   10. Phân tích kết quả      -> src/models/evaluate.py (sai số theo nhóm, SHAP, biểu đồ)

Chạy:  python -m src.models.train      (~1 phút trên máy 4 nhân)
"""
import json
import time
from datetime import datetime

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.data.loader import clean_train, load_store, load_train, split_by_time
from src.features.pipeline import RossmannPipeline, inverse_target, target_log
from src.models.evaluate import (BaselineA, baseline_b_predict, evaluate_predictions, fit_interval,
                                 interval_coverage)
from src.models.model_loader import save_artifact

MODEL_VERSION = "1.0.0"
FEATURE_SET = "base"

# Tham số giữ như notebook gốc (chưa tinh chỉnh). Random Forest dùng cấu hình nhẹ
# (50 cây, mỗi cây học 10% dữ liệu) để train được trên máy cá nhân.
# Linear Regression: Store, thứ, tháng, năm là mã số (không có thứ tự) nên được mã hoá one-hot,
# các đặc trưng còn lại đưa vào trực tiếp (giống hồi quy đa biến trong bài học).
ONE_HOT_COLUMNS = ["Store", "DayOfWeek", "Month", "Year"]
PARAMS = {
    "Linear Regression": {"one_hot_columns": ONE_HOT_COLUMNS},
    "Decision Tree": {"max_depth": 15, "min_samples_leaf": 10, "random_state": config.SEED},
    "Random Forest": {"n_estimators": 50, "max_depth": 15, "min_samples_leaf": 10,
                      "max_samples": 0.1, "random_state": config.SEED, "n_jobs": -1},
    "XGBoost": {"objective": "reg:squarederror", "learning_rate": 0.1, "max_depth": 7,
                "subsample": 0.9, "colsample_bytree": 0.9, "n_estimators": 400,
                "random_state": config.SEED, "n_jobs": -1},
    "LightGBM": {"objective": "regression", "learning_rate": 0.1, "max_depth": 8,
                 "num_leaves": 63, "n_estimators": 400, "random_state": config.SEED, "verbose": -1},
}
BASELINE_A = "Baseline A (Store-DOW-Promo)"
BASELINE_B = "Baseline B (Seasonal Naive 52w)"
PROPHET_FILE = config.OUT_EVAL / "prophet_metrics.json"     # sinh bởi scripts/prophet_experiment.py


def make_model(name, params=None):
    params = params or PARAMS[name]
    if name == "Linear Regression":
        one_hot = ColumnTransformer([("one_hot", OneHotEncoder(handle_unknown="ignore"), params["one_hot_columns"])],
                                    remainder="passthrough")
        return make_pipeline(one_hot, LinearRegression())
    if name == "Decision Tree":
        return DecisionTreeRegressor(**params)
    if name == "Random Forest":
        return RandomForestRegressor(**params)
    if name == "XGBoost":
        return xgb.XGBRegressor(**params)
    if name == "LightGBM":
        return lgb.LGBMRegressor(**params)
    raise ValueError(f"Không có mô hình {name}")


def metric_row(name, v, t, seconds, selected):
    row = {"Model": name}
    for split, m in [("Val", v), ("Test", t)]:
        for metric in ["RMSPE", "MAE", "MSE", "RMSE", "R2"]:
            row[f"{split} {metric}"] = m[metric]
    row.update({"Train seconds": seconds, "Selected": selected})
    return row


def main():
    t0 = time.time()
    for d in [config.MODELS_DIR, config.OUT_EVAL, config.OUT_FIG, config.OUT_PRED]:
        d.mkdir(parents=True, exist_ok=True)

    # 1–3. Load, làm sạch, chia theo thời gian
    clean = clean_train(load_train())
    train, val, test = split_by_time(clean)
    print(f"Train {len(train):,} | Validation {len(val):,} | Test {len(test):,}")

    # 4. Feature Engineering – pipeline chỉ học từ Train
    pipe = RossmannPipeline(FEATURE_SET).fit(train, load_store())
    X_train, X_val, X_test = pipe.transform(train), pipe.transform(val), pipe.transform(test)
    y_train = target_log(train.Sales)

    # 5. Baseline (học từ Train)
    base_a = BaselineA().fit(train)
    preds = {BASELINE_A: (base_a.predict(val), base_a.predict(test))}
    bb_val, fb_val = baseline_b_predict(val, clean, base_a)
    bb_test, fb_test = baseline_b_predict(test, clean, base_a)
    preds[BASELINE_B] = (bb_val, bb_test)

    # 6. Train các mô hình học máy
    models, seconds = {}, {}
    for name in PARAMS:
        start = time.time()
        models[name] = make_model(name).fit(X_train, y_train)
        seconds[name] = round(time.time() - start, 1)
        preds[name] = (inverse_target(models[name].predict(X_val)), None)
        print(f"  {name:13s} Val RMSPE {evaluate_predictions(val.Sales, preds[name][0])['RMSPE']:.6f} "
              f"({seconds[name]}s)")

    # 7. Chọn mô hình theo RMSPE Validation (chỉ trong các mô hình học máy)
    val_rmspe = {name: evaluate_predictions(val.Sales, preds[name][0])["RMSPE"] for name in PARAMS}
    best = min(val_rmspe, key=val_rmspe.get)
    print(f">> Chọn theo Validation: {best} (Val RMSPE {val_rmspe[best]:.6f})")

    # 8. Test – chỉ tính sau khi đã chọn
    for name in PARAMS:
        preds[name] = (preds[name][0], inverse_target(models[name].predict(X_test)))
    rows = []
    for name, (p_val, p_test) in preds.items():
        v, t = evaluate_predictions(val.Sales, p_val), evaluate_predictions(test.Sales, p_test)
        rows.append(metric_row(name, v, t, seconds.get(name), name == best))
    if PROPHET_FILE.exists():   # Prophet: mô hình chuỗi thời gian theo từng cửa hàng (chỉ để so sánh)
        prophet = json.loads(PROPHET_FILE.read_text(encoding="utf-8"))
        rows.append(metric_row("Prophet (từng cửa hàng)", prophet["val"], prophet["test"], prophet.get("seconds"), False))
    table = pd.DataFrame(rows)
    table.to_csv(config.OUT_EVAL / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(table.drop(columns="Train seconds").round(4).to_string(index=False))

    # 9. Lưu model được chọn + metadata
    pred_val, pred_test = preds[best]
    interval = fit_interval(val.Sales, pred_val)
    interval["test_coverage"] = interval_coverage(test.Sales, pred_test, interval)
    metadata = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": best,
        "feature_set": FEATURE_SET,
        "features": pipe.features,
        "categorical_features": pipe.categorical,
        "encoder_categories": {c: [str(x) for x in cats]
                               for c, cats in zip(pipe.categorical, pipe.encoder.categories_, strict=True)},
        "hyperparameters": PARAMS[best],
        "models_compared": ["Baseline A", "Baseline B", *PARAMS] + (["Prophet"] if PROPHET_FILE.exists() else []),
        "target": "log1p(Sales); dự báo = max(expm1(pred), 0)",
        "train_period": pipe.train_date_range,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "cleaning_rule": "Open != 0 and Sales > 0",
        "selection_rule": ("RMSPE Validation thấp nhất trong các mô hình hồi quy dùng chung cho mọi cửa hàng; "
                           "Prophet (mỗi cửa hàng 1 mô hình) chỉ để so sánh; Test chỉ dùng để báo cáo"),
        "metrics": {"val": evaluate_predictions(val.Sales, pred_val),
                    "test": evaluate_predictions(test.Sales, pred_test)},
        "baseline_a_metrics": {"val": evaluate_predictions(val.Sales, preds[BASELINE_A][0]),
                               "test": evaluate_predictions(test.Sales, preds[BASELINE_A][1])},
        "baseline_b_fallback_rows": {"val": fb_val, "test": fb_test},
        "prediction_interval": interval,
        "median_competition_distance": pipe.median_competition_distance,
        "note": "Model chỉ học từ tập Train (<= 2015-05-07) nên metric Test áp dụng đúng cho file này.",
    }
    save_artifact(models[best], pipe, base_a, metadata)
    (config.OUT_EVAL / "train_run_summary.json").write_text(json.dumps(
        {"selected": best, "total_seconds": round(time.time() - t0, 1), "interval": interval},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu model {best} ({time.time() - t0:.0f}s). Đang phân tích kết quả…")

    # 10. Phân tích kết quả của model đã lưu (nạp lại từ file -> kiểm tra luôn save/load)
    from src.models.evaluate import main as evaluate_main
    evaluate_main()


if __name__ == "__main__":
    main()
