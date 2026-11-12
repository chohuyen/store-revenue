"""PlanningService – lập kế hoạch nguồn lực từ doanh thu dự báo.

Các bước (mỗi bước là một hàm nhỏ):
    1. calendar()    Lấy lịch kỳ tới của các cửa hàng từ test.csv (Kaggle, 01/08 – 17/09/2015):
                     có sẵn Open, Promo, StateHoliday, SchoolHoliday thật cho 856 cửa hàng.
    2. forecast()    Dự báo doanh thu từng ngày bằng model đã lưu (không train lại) + khoảng 80%.
    3. staffing()    Quy đổi doanh thu ra số ca nhân viên:
                         số ca = ceil(doanh thu ÷ doanh thu 1 ca), tối thiểu `min_shifts` khi mở cửa,
                         0 ca khi đóng cửa. Nếu nhập tổng ca có sẵn/ngày thì phân bổ ca giữa các
                         cửa hàng theo tỷ trọng doanh thu dự báo.
    4. aggregate()   Tổng hợp theo Ngày / Tuần / Tháng.
    5. ranking()     Xếp hạng cửa hàng theo doanh thu dự báo, so với cùng kỳ năm trước.
    6. suggestions() Gợi ý hành động theo các quy tắc đơn giản, mỗi gợi ý ghi rõ căn cứ.

"Doanh thu 1 ca" là GIẢ ĐỊNH do người dùng nhập vì dữ liệu Rossmann không có số liệu nhân sự.
Gợi ý là hỗ trợ ra quyết định, không phải lời giải tối ưu.
"""
import numpy as np
import pandas as pd

from src.data.loader import load_test

FREQ = {"Ngày": "D", "Tuần": "W", "Tháng": "M"}
PEAK_RATIO = 1.2          # ngày có dự báo > 120% trung bình kỳ của cửa hàng -> cao điểm
PROMO_UPLIFT = 0.20       # khuyến mãi làm dự báo tăng > 20% -> gợi ý cân nhắc khuyến mãi
YOY_DROP = -0.10          # dự báo giảm > 10% so với cùng kỳ năm trước -> cảnh báo


class PlanningService:
    def __init__(self, prediction_service, history=None):
        self.prediction_service = prediction_service
        self.history = history        # doanh thu ngày lịch sử (DashboardService.data) để so cùng kỳ
        self._calendar = None

    # ------------------------------------------------------------------ 1. lịch kỳ tới
    def full_calendar(self):
        if self._calendar is None:
            self._calendar = load_test()[["Store", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]]
        return self._calendar

    def available_stores(self):
        return sorted(self.full_calendar()["Store"].unique().tolist())

    def date_bounds(self):
        cal = self.full_calendar()
        return cal["Date"].min(), cal["Date"].max()

    def calendar(self, stores, start, end):
        if not stores:
            raise ValueError("Hãy nhập ít nhất một mã cửa hàng")
        if len(stores) > 50:
            raise ValueError("Tối đa 50 cửa hàng cho một lần lập kế hoạch")
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        lo, hi = self.date_bounds()
        if start > end:
            raise ValueError("'Từ ngày' phải nhỏ hơn hoặc bằng 'Đến ngày'")
        if end < lo or start > hi:
            raise ValueError(f"Lịch kỳ tới chỉ có từ {lo:%Y-%m-%d} đến {hi:%Y-%m-%d}")
        cal = self.full_calendar()
        missing = sorted(set(stores) - set(cal["Store"]))
        if missing:
            raise ValueError(f"Không có lịch kỳ tới cho cửa hàng {missing[:10]} "
                             f"(test.csv chỉ có {cal['Store'].nunique()} cửa hàng)")
        sel = cal[cal["Store"].isin(stores) & cal["Date"].between(start, end)]
        return sel.sort_values(["Date", "Store"]).reset_index(drop=True)

    # ------------------------------------------------------------------ 2. dự báo
    def forecast(self, cal):
        out, warnings = self.prediction_service.predict_frame(cal)
        out["Date"] = pd.to_datetime(out["Date"])
        out["Open"] = cal["Open"].to_numpy()
        closed = out["Open"] == 0
        out.loc[closed, ["PredLow", "PredHigh", "ExpectedSales"]] = 0.0
        out.loc[closed, "LowForecastAlert"] = 0
        return out, warnings

    # ------------------------------------------------------------------ 3. quy đổi ra số ca
    @staticmethod
    def staffing(fc, revenue_per_shift, min_shifts=2, basis="PredictedSales", shift_budget=0):
        if revenue_per_shift <= 0:
            raise ValueError("Doanh thu 1 ca phải lớn hơn 0")
        fc = fc.copy()
        fc["PlanSales"] = fc[basis]
        need = np.ceil(fc["PlanSales"] / revenue_per_shift)
        fc["ShiftsNeeded"] = np.where(fc["Open"] == 1, np.maximum(need, min_shifts), 0).astype(int)
        if shift_budget and shift_budget > 0:
            fc["ShiftsAllocated"] = 0
            for _, idx in fc.groupby("Date").groups.items():
                fc.loc[idx, "ShiftsAllocated"] = allocate(fc.loc[idx, "PlanSales"], fc.loc[idx, "ShiftsNeeded"],
                                                          shift_budget)
        return fc

    # ------------------------------------------------------------------ 4. tổng hợp theo kỳ
    @staticmethod
    def aggregate(plan, granularity="Tuần"):
        freq = FREQ[granularity]
        period = plan["Date"].dt.to_period(freq).dt.start_time
        cols = [c for c in ["PredictedSales", "PredLow", "PredHigh", "ShiftsNeeded", "ShiftsAllocated"] if c in plan]
        agg = plan.assign(Period=period).groupby(["Period", "Store"])[cols].sum().reset_index()
        agg["OpenDays"] = plan.assign(Period=period).groupby(["Period", "Store"])["Open"].sum().to_numpy()
        return agg

    # ------------------------------------------------------------------ 5. xếp hạng cửa hàng
    def ranking(self, plan):
        g = plan.groupby("Store").agg(ForecastSales=("PredictedSales", "sum"), OpenDays=("Open", "sum"),
                                      ShiftsNeeded=("ShiftsNeeded", "sum")).reset_index()
        g["AvgPerOpenDay"] = g["ForecastSales"] / g["OpenDays"].replace(0, np.nan)
        g["LastYearSales"] = [self._last_year(s, plan) for s in g["Store"]]
        g["ChangeVsLastYear_%"] = np.where(g["LastYearSales"] > 0,
                                           (g["ForecastSales"] / g["LastYearSales"] - 1) * 100, np.nan)
        g = g.sort_values("ForecastSales", ascending=False).reset_index(drop=True)
        g.insert(0, "Rank", np.arange(1, len(g) + 1))
        return g

    def _last_year(self, store, plan):
        """Doanh thu thực tế cùng kỳ năm trước (cùng thứ, lùi 52 tuần); NaN nếu không có lịch sử."""
        if self.history is None:
            return np.nan
        start = plan["Date"].min() - pd.Timedelta(weeks=52)
        end = plan["Date"].max() - pd.Timedelta(weeks=52)
        h = self.history
        h = h[(h["Store"] == store) & h["Date"].between(start, end)]
        return float(h["Sales"].sum()) if len(h) else np.nan

    # ------------------------------------------------------------------ 6. gợi ý
    def suggestions(self, plan, ranking):
        rows = []
        for store, p in plan.groupby("Store"):
            open_days = p[p["Open"] == 1]
            if open_days.empty:
                continue
            avg = open_days["PredictedSales"].mean()
            peak = open_days[open_days["PredictedSales"] > PEAK_RATIO * avg]
            if len(peak):
                rows.append(_s(store, _days(peak), "Tăng ca ngày cao điểm",
                               f"{len(peak)} ngày dự báo cao hơn {PEAK_RATIO:.0%} mức trung bình kỳ "
                               f"({avg:,.0f} €/ngày); cần tới {int(peak['ShiftsNeeded'].max())} ca/ngày."))
            low = open_days[open_days["LowForecastAlert"] == 1]
            if len(low):
                rows.append(_s(store, _days(low), "Xem lại vận hành",
                               f"{len(low)} ngày dự báo thấp hơn kỳ vọng lịch sử cùng thứ/khuyến mãi "
                               f"(kỳ vọng > cận trên khoảng 80%)."))
            if "ShiftsAllocated" in p and (p["ShiftsAllocated"] < p["ShiftsNeeded"]).any():
                short = p[p["ShiftsAllocated"] < p["ShiftsNeeded"]]
                rows.append(_s(store, _days(short), "Thiếu ca so với nhu cầu",
                               f"Tổng ca có sẵn không đủ: thiếu {int((short['ShiftsNeeded'] - short['ShiftsAllocated']).sum())} "
                               f"ca trong {len(short)} ngày."))
            uplift = self._promo_uplift(open_days)
            if uplift is not None and len(uplift):
                rows.append(_s(store, _days(uplift), "Cân nhắc khuyến mãi",
                               f"{len(uplift)} ngày thường chưa khuyến mãi; mô phỏng What-if cho thấy doanh thu dự báo "
                               f"tăng trung bình {uplift['Uplift'].mean():.0%} nếu có khuyến mãi "
                               f"(mô phỏng, chưa tính chi phí khuyến mãi)."))
            r = ranking[ranking["Store"] == store].iloc[0]
            if pd.notna(r["ChangeVsLastYear_%"]) and r["ChangeVsLastYear_%"] < YOY_DROP * 100:
                rows.append(_s(store, "Cả kỳ", "Doanh thu giảm so với cùng kỳ",
                               f"Dự báo {r['ForecastSales']:,.0f} € thấp hơn {abs(r['ChangeVsLastYear_%']):.1f}% "
                               f"so với cùng kỳ năm trước ({r['LastYearSales']:,.0f} €)."))
            closed = int((p["Open"] == 0).sum())
            if closed:
                rows.append(_s(store, _days(p[p["Open"] == 0]), "Ngày đóng cửa",
                               f"{closed} ngày đóng cửa theo lịch: không bố trí ca."))
        return pd.DataFrame(rows, columns=["Store", "Ngày", "Gợi ý", "Căn cứ"])

    def _promo_uplift(self, open_days):
        """What-if Promo 0 -> 1 cho các ngày thường (T2–T6) chưa có khuyến mãi; dùng model đã lưu, không train."""
        cand = open_days[(open_days["Promo"] == 0) & (open_days["Date"].dt.dayofweek < 5)]
        if cand.empty:
            return None
        sim, _ = self.prediction_service.predict_frame(
            cand[["Store", "Date", "StateHoliday", "SchoolHoliday"]].assign(Promo=1))
        cand = cand.assign(Uplift=sim["PredictedSales"].to_numpy() / cand["PredictedSales"].to_numpy() - 1)
        return cand[cand["Uplift"] > PROMO_UPLIFT]

    # ------------------------------------------------------------------ chạy trọn gói
    def run(self, stores, start, end, revenue_per_shift, min_shifts=2, basis="PredictedSales",
            shift_budget=0, granularity="Tuần"):
        cal = self.calendar(stores, start, end)
        fc, warnings = self.forecast(cal)
        plan = self.staffing(fc, revenue_per_shift, min_shifts, basis, shift_budget)
        rank = self.ranking(plan)
        return {"plan": plan, "summary": self.aggregate(plan, granularity), "ranking": rank,
                "suggestions": self.suggestions(plan, rank), "warnings": warnings}


def allocate(sales, needed, budget):
    """Chia `budget` ca giữa các cửa hàng mở cửa theo tỷ trọng doanh thu (phương pháp phần dư lớn nhất),
    không vượt quá số ca cần của từng cửa hàng."""
    sales, needed = np.asarray(sales, float), np.asarray(needed, int)
    if needed.sum() <= budget:
        return needed
    share = np.where(needed > 0, sales, 0.0)
    if share.sum() == 0:
        return np.zeros_like(needed)
    raw = budget * share / share.sum()
    alloc = np.minimum(np.floor(raw).astype(int), needed)
    left = budget - alloc.sum()
    for i in np.argsort(-(raw - np.floor(raw))):
        if left <= 0:
            break
        if alloc[i] < needed[i]:
            alloc[i] += 1
            left -= 1
    return alloc


def _days(frame):
    dates = sorted(frame["Date"].dt.strftime("%d/%m"))
    return ", ".join(dates[:6]) + (f" … (+{len(dates) - 6})" if len(dates) > 6 else "")


def _s(store, days, kind, reason):
    return {"Store": int(store), "Ngày": days, "Gợi ý": kind, "Căn cứ": reason}

