"""Phân tích khám phá dữ liệu (EDA) – sinh hình và số liệu cho Chương 3 của báo cáo.
Chạy: python scripts/eda.py  -> results/figures/eda_*.png, results/evaluation/eda_summary.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data.loader import clean_train, load_store, load_train  # noqa: E402

BLUE, TEAL, AMBER = "#1f6feb", "#12b5a6", "#f59e0b"
DAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def save(fig, name):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / name, dpi=130)
    plt.close(fig)


def main():
    config.OUT_FIG.mkdir(parents=True, exist_ok=True)
    raw = load_train()
    store = load_store()
    df = clean_train(raw).merge(store, on="Store", how="left")
    s = {"rows_raw": len(raw), "rows_clean": len(df), "stores": int(df.Store.nunique()),
         "date_range": [str(df.Date.min().date()), str(df.Date.max().date())],
         "closed_rows_open0": int((raw.Open == 0).sum()), "open_but_zero_sales": int(((raw.Open == 1) & (raw.Sales == 0)).sum()),
         "sales_mean": float(df.Sales.mean()), "sales_median": float(df.Sales.median()),
         "skew_sales": float(df.Sales.skew()), "skew_log1p_sales": float(np.log1p(df.Sales).skew())}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(df.Sales, bins=60, color=BLUE)
    ax[0].set_title(f"Sales (skew {s['skew_sales']:.2f})")
    ax[1].hist(np.log1p(df.Sales), bins=60, color=AMBER)
    ax[1].set_title(f"log1p(Sales) (skew {s['skew_log1p_sales']:.2f}) → dùng làm target")
    save(fig, "eda_1_sales_distribution.png")

    dow = df.groupby("DayOfWeek").Sales.agg(["mean", "size"])
    s["avg_sales_by_dow"] = dow["mean"].round(0).to_dict()
    s["rows_by_dow"] = dow["size"].to_dict()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(DAYS, dow["mean"], color=BLUE)
    ax.set_title("Doanh thu trung bình theo thứ (CN chỉ có ít cửa hàng mở)")
    save(fig, "eda_2_sales_by_dayofweek.png")

    monthly = df.groupby([df.Date.dt.year.rename("Year"), df.Date.dt.month.rename("Month")]).Sales.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    for year in monthly.columns:
        ax.plot(monthly.index, monthly[year], marker="o", label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_title("Doanh thu TB / cửa hàng / ngày theo tháng")
    ax.legend()
    save(fig, "eda_3_monthly_seasonality.png")

    promo = df.groupby("Promo").Sales.mean()
    s["avg_sales_promo"] = {int(k): float(v) for k, v in promo.items()}
    s["promo_uplift_pct_observed"] = float((promo[1] / promo[0] - 1) * 100)
    s["promo_on_weekend_rows"] = int(((raw.DayOfWeek >= 6) & (raw.Promo == 1)).sum())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(["Không KM", "Có KM"], promo.values, color=[BLUE, AMBER])
    ax[0].set_title(f"Promo (chênh lệch quan sát {s['promo_uplift_pct_observed']:.1f}%)")
    st = df.groupby("StoreType").Sales.mean()
    ax[1].bar(st.index, st.values, color=TEAL)
    ax[1].set_title("Theo StoreType")
    asm = df.groupby("Assortment").Sales.mean()
    ax[2].bar(asm.index, asm.values, color=TEAL)
    ax[2].set_title("Theo Assortment (a basic, b extra, c extended)")
    save(fig, "eda_4_promo_storetype_assortment.png")

    sh = df.groupby("StateHoliday").Sales.agg(["mean", "size"])
    s["stateholiday_rows_with_sales"] = sh["size"].to_dict()
    s["avg_sales_by_stateholiday"] = sh["mean"].round(0).to_dict()
    bins = pd.cut(df.CompetitionDistance, [0, 500, 1000, 2000, 5000, 10000, np.inf],
                  labels=["<0.5km", "0.5–1km", "1–2km", "2–5km", "5–10km", ">10km"])
    comp = df.groupby(bins, observed=True).Sales.mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(sh.index, sh["mean"], color=AMBER)
    ax[0].set_title("Theo StateHoliday (a/b/c rất ít dòng)")
    ax[1].bar(comp.index.astype(str), comp.values, color=BLUE)
    ax[1].set_title("Theo khoảng cách tới đối thủ")
    save(fig, "eda_5_holiday_competition.png")

    cols = ["Sales", "Customers", "Promo", "SchoolHoliday", "DayOfWeek", "CompetitionDistance", "Promo2"]
    corr = df[cols].corr()
    s["corr_with_sales"] = corr["Sales"].drop("Sales").round(3).to_dict()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Tương quan Pearson\n(Customers chỉ để phân tích, KHÔNG dùng làm đặc trưng)", fontsize=10)
    save(fig, "eda_6_correlation.png")

    (config.OUT_EVAL / "eda_summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
    print(json.dumps(s, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
