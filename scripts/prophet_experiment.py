"""Prophet – mô hình chuỗi thời gian (Meta) để so sánh với các mô hình hồi quy.

Cách làm: mỗi cửa hàng một mô hình Prophet riêng (1.115 mô hình)
    y = log1p(Sales); mùa vụ theo tuần và theo năm; biến bổ sung: Promo, SchoolHoliday, IsStateHoliday.
    Chỉ học từ tập Train (<= 2015-05-07), dự báo cho Validation và Test giống các mô hình khác.
Kết quả: results/evaluation/prophet_metrics.json (train.py đọc file này để thêm Prophet vào bảng so sánh).

Prophet KHÔNG cần để chạy ứng dụng; chỉ cài khi muốn chạy lại thí nghiệm:
    pip install prophet
    python scripts/prophet_experiment.py        (~3–6 phút)
"""
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.loader import clean_train, load_train, split_by_time  # noqa: E402
from src.models.evaluate import evaluate_predictions  # noqa: E402

REGRESSORS = ["Promo", "SchoolHoliday", "IsStateHoliday"]


def prepare(df):
    out = df[["Store", "Date", "Sales"] + REGRESSORS[:2]].rename(columns={"Date": "ds"})
    out["IsStateHoliday"] = (df["StateHoliday"] != "0").astype(int).to_numpy()
    out["y"] = np.log1p(out["Sales"])
    return out


def fit_one_store(args):
    """Train Prophet cho 1 cửa hàng, trả về dự báo (€) cho các ngày cần dự báo."""
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    from prophet import Prophet
    train, future = args
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    for reg in REGRESSORS:
        model.add_regressor(reg)
    model.fit(train[["ds", "y"] + REGRESSORS])
    future = future.sort_values("ds")          # predict() trả kết quả theo thứ tự ngày tăng dần
    pred = model.predict(future[["ds"] + REGRESSORS])["yhat"].to_numpy()
    return future.index.to_numpy(), np.maximum(np.expm1(pred), 0)


def main():
    start = time.time()
    train, val, test = split_by_time(clean_train(load_train()))
    tr, target = prepare(train), prepare(pd.concat([val, test]))
    target = target.reset_index(drop=True)
    jobs = [(tr[tr.Store == s], target[target.Store == s]) for s in sorted(target.Store.unique())]
    pred = np.zeros(len(target))
    with Pool() as pool:
        for i, (idx, p) in enumerate(pool.imap_unordered(fit_one_store, jobs, chunksize=8), 1):
            pred[idx] = p
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} cửa hàng")
    is_val = target["ds"] <= config.VAL_END
    result = {
        "model": "Prophet (mỗi cửa hàng 1 mô hình)",
        "settings": {"target": "log1p(Sales)", "seasonality": ["weekly", "yearly"], "regressors": REGRESSORS},
        "n_stores": len(jobs),
        "val": evaluate_predictions(target.loc[is_val, "Sales"], pred[is_val]),
        "test": evaluate_predictions(target.loc[~is_val, "Sales"], pred[~is_val]),
        "seconds": round(time.time() - start, 1),
    }
    config.OUT_EVAL.mkdir(parents=True, exist_ok=True)
    (config.OUT_EVAL / "prophet_metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["val", "test", "seconds"]}, indent=1))


if __name__ == "__main__":
    main()
