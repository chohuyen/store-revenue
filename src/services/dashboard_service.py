"""DashboardService: KPI và biểu đồ tổng quan từ dữ liệu lịch sử THẬT (train.csv đã làm sạch).

Dữ liệu ngày được gom theo Store–Tháng–Promo (cột Sales = tổng, Rows = số dòng cửa hàng-ngày),
mọi KPI tính từ bảng này nên nhanh và nhất quán. Bộ lọc thời gian làm việc theo THÁNG.
"""
import pandas as pd

from src.data.loader import clean_train, load_store, load_train

MONTHLY_COLUMNS = ["Month", "Store", "StoreType", "Assortment", "Promo", "Sales", "Rows"]


def build_monthly(daily):
    """daily: dữ liệu sạch có Store, Date, Sales, Promo, StoreType, Assortment."""
    d = daily.assign(Month=daily["Date"].dt.to_period("M").dt.start_time)
    g = (d.groupby(["Month", "Store", "StoreType", "Assortment", "Promo"])["Sales"]
         .agg(Sales="sum", Rows="count").reset_index())
    return g[MONTHLY_COLUMNS]


def load_daily():
    """Đọc train.csv (tự giải nén datasets/train.zip nếu cần), làm sạch, ghép StoreType/Assortment."""
    data = clean_train(load_train())[["Store", "Date", "Sales", "Promo"]]
    return data.merge(load_store()[["Store", "StoreType", "Assortment"]], on="Store", how="left")


class DashboardService:
    def __init__(self, daily=None):
        self.data = load_daily() if daily is None else daily     # dữ liệu ngày
        self.monthly = build_monthly(self.data)                    # dữ liệu gom theo tháng

    # ------------------------------------------------------------------ lọc
    def filter(self, start=None, end=None, store_type=None):
        m = self.monthly
        if start is not None:
            m = m[m["Month"] >= pd.Timestamp(start).to_period("M").start_time]
        if end is not None:
            m = m[m["Month"] <= pd.Timestamp(end).to_period("M").start_time]
        if store_type and store_type != "Tất cả":
            m = m[m["StoreType"] == store_type]
        if m.empty:
            raise ValueError("Không có dữ liệu trong phạm vi đã chọn")
        return m

    # ------------------------------------------------------------------ chỉ số
    def kpis(self, start=None, end=None, store_type=None):
        m = self.filter(start, end, store_type)
        promo = m.groupby("Promo")[["Sales", "Rows"]].sum()
        avg = (promo["Sales"] / promo["Rows"]).to_dict()
        return {
            "start": f"{m['Month'].min():%Y-%m}", "end": f"{m['Month'].max():%Y-%m}",
            "stores": int(m["Store"].nunique()), "months": int(m["Month"].nunique()),
            "rows": int(m["Rows"].sum()),
            "total_sales": float(m["Sales"].sum()),
            "avg_sales_per_store_day": float(m["Sales"].sum() / m["Rows"].sum()),
            "avg_sales_promo": float(avg.get(1, float("nan"))),
            "avg_sales_no_promo": float(avg.get(0, float("nan"))),
        }

    def trend(self, start=None, end=None, store_type=None):
        """Tổng doanh thu theo tháng."""
        m = self.filter(start, end, store_type)
        return m.groupby("Month")["Sales"].sum().reset_index().rename(columns={"Month": "Date"})

    def top_stores(self, n=10, start=None, end=None, store_type=None):
        m = self.filter(start, end, store_type)
        g = m.groupby(["Store", "StoreType", "Assortment"])[["Sales", "Rows"]].sum().reset_index()
        g["AvgDailySales"] = g["Sales"] / g["Rows"]
        g = g.rename(columns={"Sales": "TotalSales", "Rows": "OpenDays"})
        g = g[["Store", "StoreType", "Assortment", "TotalSales", "AvgDailySales", "OpenDays"]]
        return g.sort_values("TotalSales", ascending=False).head(n).reset_index(drop=True)

    def compare_stores(self, stores, start=None, end=None):
        m = self.filter(start, end)
        m = m[m["Store"].isin([int(s) for s in stores])]
        if m.empty:
            raise ValueError("Không có dữ liệu cho các cửa hàng đã chọn")
        return m.groupby(["Month", "Store"])["Sales"].sum().unstack("Store")

    def store_history(self, store, before, days=42):
        """Doanh thu ngày của 1 cửa hàng trong `days` ngày trước `before`."""
        d = self.data
        before = pd.Timestamp(before)
        return d[(d.Store == int(store)) & (d.Date < before) & (d.Date >= before - pd.Timedelta(days=days))]

