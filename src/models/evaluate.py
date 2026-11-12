"""ML Layer – đánh giá: metric, baseline, khoảng dự báo, phân tích sai số, giải thích mô hình.

Metric:
    RMSPE = sqrt(mean(((y - ŷ) / y)^2))  – sai số phần trăm, metric chính của cuộc thi Rossmann,
            chỉ tính trên dòng Sales != 0 (đã lọc Sales > 0 nên dùng toàn bộ dòng; n_rmspe ghi lại số dòng).
    MAE    = mean(|y - ŷ|)               – sai số tuyệt đối trung bình (€).
    MSE    = mean((y - ŷ)^2)             – sai số bình phương trung bình.
    RMSE   = sqrt(MSE)                   – phạt nặng các lỗi lớn (€).
    R²     = 1 - SSE/SST                 – tỷ lệ biến động của doanh thu được mô hình giải thích.

Chạy:  python -m src.models.evaluate   (nạp model đã lưu -> bảng, hình trong results/)
"""
import json

import matplotlib
import numpy as np
import pandas as pd

from src import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, TEAL, AMBER, GREY = "#1f6feb", "#12b5a6", "#f59e0b", "#9aa5b1"
DAY_LABELS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


# ------------------------------------------------------------------ metric
def rmspe(y_true, y_pred):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y != 0
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(((y[mask] - p[mask]) / y[mask]) ** 2)))


def evaluate_predictions(y_true, y_pred):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    mse = float(np.mean((y - p) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        "RMSPE": rmspe(y, p),
        "MAE": float(np.mean(np.abs(y - p))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else float("nan"),
        "n_rows": int(len(y)),
        "n_rmspe": int((y != 0).sum()),
    }


# ------------------------------------------------------------------ baseline
class BaselineA:
    """Trung bình Sales theo (Store, DayOfWeek, Promo) học từ Train. Fallback = trung bình Train."""

    def fit(self, train):
        self.table = (train.groupby(["Store", "DayOfWeek", "Promo"])["Sales"].mean()
                      .rename("BaselineA").reset_index())
        self.global_mean = float(train["Sales"].mean())
        return self

    def predict(self, df):
        keys = df[["Store", "DayOfWeek", "Promo"]].astype(int)
        m = keys.merge(self.table, on=["Store", "DayOfWeek", "Promo"], how="left")
        return m["BaselineA"].fillna(self.global_mean).to_numpy()


def baseline_b_predict(split, history, baseline_a):
    """Seasonal naive: Sales cùng Store 52 tuần trước (cùng thứ). Thiếu -> Baseline A.
    `history` là dữ liệu sạch đã biết; với Val/Test, ngày lag luôn trước ngày bắt đầu tập."""
    lag = history[["Store", "Date", "Sales"]].copy()
    lag["Date"] = lag["Date"] + pd.Timedelta(weeks=52)
    m = split[["Store", "Date"]].merge(lag.rename(columns={"Sales": "lag"}), on=["Store", "Date"], how="left")
    p = m["lag"].to_numpy(float)
    fallback = np.isnan(p)
    p[fallback] = baseline_a.predict(split)[fallback]
    return p, int(fallback.sum())


# ------------------------------------------------------------------ khoảng dự báo
def fit_interval(y_val, pred_val, lower_q=0.10, upper_q=0.90):
    """Khoảng dự báo thực nghiệm: tỷ lệ Actual/Pred trên Validation, lấy phân vị 10% và 90%.
    Dự báo mới: [pred * q_low, pred * q_high] (≈ 80% nếu phân phối sai số ổn định)."""
    ratio = np.asarray(y_val, float) / np.maximum(np.asarray(pred_val, float), 1.0)
    return {"lower_q": lower_q, "upper_q": upper_q,
            "ratio_low": float(np.quantile(ratio, lower_q)),
            "ratio_high": float(np.quantile(ratio, upper_q)),
            "fitted_on": "validation"}


def interval_coverage(y, pred, interval):
    lo, hi = pred * interval["ratio_low"], pred * interval["ratio_high"]
    y = np.asarray(y, float)
    return float(np.mean((y >= lo) & (y <= hi)))


# ------------------------------------------------------------------ sai số theo nhóm
def error_by_group(frame, col):
    """frame cần cột Sales, Pred và cột nhóm."""
    rows = []
    for key, g in frame.groupby(col, observed=True):
        m = evaluate_predictions(g["Sales"], g["Pred"])
        rows.append({col: key, "n": m["n_rows"], "RMSPE": m["RMSPE"], "MAE": m["MAE"],
                     "Bias_%": float((g["Pred"].sum() / g["Sales"].sum() - 1) * 100)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ giải thích mô hình (SHAP)
def shap_importance(model, X, max_rows=20000, seed=config.SEED):
    """Mean |SHAP| của từng đặc trưng (thang log1p). XGBoost/LightGBM tính SHAP sẵn
    (pred_contribs), không cần thư viện shap. Mô hình khác -> feature_importances_."""
    Xs = X.sample(min(max_rows, len(X)), random_state=seed)
    if hasattr(model, "get_booster"):
        import xgboost as xgb
        contrib = model.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)[:, :-1]
    elif hasattr(model, "booster_"):
        contrib = model.predict(Xs, pred_contrib=True)[:, :-1]
    else:
        return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False), None
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)
    return imp, pd.DataFrame(contrib, columns=X.columns, index=Xs.index)


# ------------------------------------------------------------------ biểu đồ
def _save(fig, name):
    config.OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / name, dpi=130)
    plt.close(fig)


def plot_model_comparison(table):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    labels = table["Model"].str.split(" (", regex=False).str[0]          # "Baseline A", "XGBoost"…
    x = np.arange(len(table))
    titles = {"RMSPE": "RMSPE (thấp hơn = tốt hơn)", "MAE": "MAE € (thấp hơn = tốt hơn)",
              "RMSE": "RMSE € (thấp hơn = tốt hơn)", "R2": "R² (cao hơn = tốt hơn)"}
    for ax, metric in zip(axes.ravel(), titles, strict=True):
        ax.bar(x - 0.2, table[f"Val {metric}"], 0.4, color=BLUE, label="Validation")
        ax.bar(x + 0.2, table[f"Test {metric}"], 0.4, color=TEAL, label="Test")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
        ax.set_title(titles[metric])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend()
    _save(fig, "fig_model_comparison.png")


def plot_actual_vs_pred(frame):
    daily = frame.groupby("Date")[["Sales", "Pred", "PredLow", "PredHigh"]].sum() / 1e6
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily["Sales"], color=BLUE, marker="o", ms=3, label="Thực tế")
    ax.plot(daily.index, daily["Pred"], color=TEAL, marker="o", ms=3, label="Dự báo")
    ax.fill_between(daily.index, daily["PredLow"], daily["PredHigh"], color=TEAL, alpha=0.15,
                    label="Khoảng dự báo 80%")
    ax.set_title("Tổng doanh thu theo ngày trên tập Test (triệu €)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    _save(fig, "fig_test_actual_vs_predicted.png")


def plot_error_distribution(frame):
    pct = (frame["Pred"] - frame["Sales"]) / frame["Sales"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pct.clip(-60, 60), bins=60, color=BLUE)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_title("Phân phối sai số % (dự báo − thực tế) / thực tế")
    axes[0].set_xlabel("%  (cắt ở ±60%)")
    sample = frame.sample(min(8000, len(frame)), random_state=config.SEED)
    axes[1].scatter(sample["Sales"], sample["Pred"], s=3, alpha=0.3, color=TEAL)
    lim = [0, sample[["Sales", "Pred"]].max().max()]
    axes[1].plot(lim, lim, color="black", lw=1)
    axes[1].set_xlabel("Thực tế (€)")
    axes[1].set_ylabel("Dự báo (€)")
    axes[1].set_title("Thực tế và dự báo (mẫu 8.000 dòng)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_test_error_distribution.png")


def plot_error_by_group(groups):
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4), sharey=True)
    for ax, (col, g) in zip(axes, groups.items(), strict=True):
        labels = DAY_LABELS if col == "DayOfWeek" else g[col].astype(str)
        ax.bar(labels, g["RMSPE"], color=AMBER)
        ax.set_title(f"RMSPE theo {col}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("RMSPE (Test)")
    _save(fig, "fig_test_rmspe_by_group.png")


def plot_shap(imp, contrib, X):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color=BLUE)
    ax.set_xlabel("Mean |SHAP| (thang log1p Sales)")
    ax.set_title("Mức ảnh hưởng của đặc trưng tới dự báo (SHAP, tập Test)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_shap_importance.png")
    if contrib is None or "Promo" not in contrib.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    promo = X.loc[contrib.index, "Promo"]
    for value, color in [(0, BLUE), (1, AMBER)]:
        ax.hist(contrib.loc[promo == value, "Promo"], bins=40, alpha=0.7, color=color, label=f"Promo = {value}")
    ax.set_xlabel("Đóng góp SHAP của Promo (thang log1p)")
    ax.set_title("Đóng góp của Promo vào dự báo")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "fig_shap_promo.png")


# ------------------------------------------------------------------ đánh giá model đã lưu
def main():
    """Nạp model ĐÃ LƯU, dự báo lại Val/Test, so với metadata (kiểm tra save/load),
    rồi xuất bảng sai số theo nhóm, SHAP và biểu đồ cho báo cáo."""
    from src.data.loader import clean_train, load_train, split_by_time
    from src.models.model_loader import load_artifact

    art = load_artifact()
    _, val, test = split_by_time(clean_train(load_train()))
    config.OUT_EVAL.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, split in [("val", val), ("test", test)]:
        result[name] = evaluate_predictions(split["Sales"], art.predict_sales(split))
        result[name]["matches_metadata"] = abs(result[name]["RMSPE"] - art.metadata["metrics"][name]["RMSPE"]) < 1e-9

    # Bảng dự báo trên Test
    prepared = art.pipeline.prepare(test)
    X_test = art.pipeline.encode(prepared)
    pred = art.predict_sales(test)
    low, high = art.interval(pred)
    frame = test[["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "Sales"]].copy()
    frame["StoreType"] = prepared["StoreType"].to_numpy()
    frame["Assortment"] = prepared["Assortment"].to_numpy()
    frame["Pred"], frame["PredLow"], frame["PredHigh"] = pred, low, high
    out = frame.rename(columns={"Pred": "PredictedSales"})
    out["AbsError"] = (out["PredictedSales"] - out["Sales"]).abs()
    out["PctError"] = out["AbsError"] / out["Sales"]
    out.to_csv(config.OUT_PRED / "test_predictions_selected_model.csv", index=False)

    # Sai số theo nhóm và theo cửa hàng
    groups = {col: error_by_group(frame, col) for col in ["DayOfWeek", "Promo", "StoreType", "Assortment"]}
    for col, g in groups.items():
        g.to_csv(config.OUT_EVAL / f"test_error_by_{col}.csv", index=False)
    store_err = frame.groupby("Store")[["Sales", "Pred"]].apply(
        lambda g: pd.Series({"n": len(g), "RMSPE": rmspe(g["Sales"], g["Pred"]),
                             "MAE": float(np.mean(np.abs(g["Sales"] - g["Pred"]))),
                             "MeanSales": float(g["Sales"].mean())}))
    store_err.sort_values("RMSPE", ascending=False).head(10).to_csv(config.OUT_EVAL / "test_error_top10_stores.csv")
    sq = ((frame["Sales"] - frame["Pred"]) ** 2).groupby(frame["Store"]).sum()
    pct = out["PctError"]
    result["test"].update({
        "interval_coverage": interval_coverage(frame["Sales"], pred, art.metadata["prediction_interval"]),
        "total_pred_million": float(pred.sum() / 1e6),
        "total_actual_million": float(frame["Sales"].sum() / 1e6),
        "bias_pct": float((pred.sum() / frame["Sales"].sum() - 1) * 100),
        "share_rows_abs_pct_error_gt_10": float((pct > 0.10).mean() * 100),
        "top10_store_share_sq_error_pct": float(sq.nlargest(10).sum() / sq.sum() * 100),
    })

    # Giải thích mô hình
    imp, contrib = shap_importance(art.model, X_test)
    imp.rename("mean_abs_shap").to_csv(config.OUT_EVAL / "shap_importance.csv", header=True)
    result["shap_top5"] = imp.head(5).round(4).to_dict()

    # Biểu đồ
    comparison = config.OUT_EVAL / "model_comparison.csv"
    if comparison.exists():
        plot_model_comparison(pd.read_csv(comparison, encoding="utf-8-sig"))
    plot_actual_vs_pred(frame)
    plot_error_distribution(frame)
    plot_error_by_group(groups)
    plot_shap(imp, contrib, X_test)

    (config.OUT_EVAL / "saved_model_evaluation.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
