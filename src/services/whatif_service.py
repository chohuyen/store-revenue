"""WhatIfService: so sánh kịch bản gốc và kịch bản thay đổi bằng model ĐÃ LƯU.

- Giữ nguyên Store, Date (và mọi thuộc tính store); chỉ đổi Promo / StateHoliday / SchoolHoliday.
- Chỉ gọi predict, KHÔNG gọi fit/train.
- Đây là mô phỏng theo mô hình dự báo, KHÔNG chứng minh tác động nhân quả của Promo hay ngày lễ.
"""
import pandas as pd

from src import config

DISCLAIMER = ("Kết quả What-if là mô phỏng dựa trên mô hình dự báo (học từ tương quan trong dữ liệu "
              "lịch sử), không chứng minh tác động nhân quả của khuyến mãi hay ngày lễ.")


def percent_change(base, new):
    """% thay đổi; trả None (hiển thị N/A) khi doanh thu gốc = 0 để tránh chia cho 0."""
    if base is None or abs(base) < 1e-9:
        return None
    return (new - base) / base * 100.0


class WhatIfService:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service

    def _row(self, store, date, promo, state_holiday, school_holiday):
        return {"Store": store, "Date": date, "Promo": promo,
                "StateHoliday": state_holiday, "SchoolHoliday": school_holiday}

    def compare(self, store, date, baseline, scenario):
        """baseline / scenario: dict có Promo, StateHoliday, SchoolHoliday."""
        rows = [self._row(store, date, baseline["Promo"], baseline["StateHoliday"], baseline["SchoolHoliday"]),
                self._row(store, date, scenario["Promo"], scenario["StateHoliday"], scenario["SchoolHoliday"])]
        out, warnings = self.prediction_service.predict_frame(pd.DataFrame(rows))
        out.insert(0, "Scenario", ["Kịch bản gốc", "Kịch bản What-if"])
        base, new = float(out.loc[0, "PredictedSales"]), float(out.loc[1, "PredictedSales"])
        return {
            "table": out,
            "baseline_sales": base,
            "scenario_sales": new,
            "abs_diff": new - base,
            "pct_diff": percent_change(base, new),
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    def scenario_grid(self, store, date, school_holiday):
        """Tất cả tổ hợp Promo {0,1} × StateHoliday {0,a,b,c} cho cùng Store/Date."""
        rows = [self._row(store, date, p, h, school_holiday)
                for h in config.STATE_HOLIDAY_VALUES for p in (0, 1)]
        out, warnings = self.prediction_service.predict_frame(pd.DataFrame(rows))
        return out, warnings
