"""PredictionService: kiểm tra đầu vào, dự báo 1 bản ghi hoặc nhiều bản ghi (CSV), xuất CSV.

Kết quả gồm: Store, Date, các input, PredictedSales, khoảng min–max (PredLow/PredHigh),
ExpectedSales (Baseline A – trung bình lịch sử cùng Store/Thứ/Promo) và cờ cảnh báo.
Nếu có cột Sales thật: ActualSales, AbsError, PctError (N/A khi Sales = 0).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import normalize_state_holiday
from src.features.pipeline import INPUT_COLUMNS

DAY_NAMES = {1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"}


class PredictionService:
    def __init__(self, model_service):
        self.model_service = model_service

    # ---------------------------------------------------------------- kiểm tra đầu vào
    def validate_input(self, df):
        """Trả về (df đã chuẩn hoá, danh sách cảnh báo). Lỗi nghiêm trọng -> ValueError."""
        if df is None or len(df) == 0:
            raise ValueError("Không có dữ liệu đầu vào")
        missing = [c for c in INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {missing}. Cần: {INPUT_COLUMNS}")
        d = df.copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        if d["Date"].isna().any():
            raise ValueError("Date không hợp lệ (định dạng YYYY-MM-DD)")
        d["StateHoliday"] = normalize_state_holiday(d["StateHoliday"])
        for col in ["Promo", "SchoolHoliday"]:
            v = pd.to_numeric(d[col], errors="coerce")
            if v.isna().any() or not v.isin([0, 1]).all():
                raise ValueError(f"{col} chỉ nhận 0 hoặc 1")
            d[col] = v.astype(int)
        s = pd.to_numeric(d["Store"], errors="coerce")
        valid = set(self.model_service.store_ids())
        bad = sorted(set(s.dropna().astype(int)) - valid)
        if s.isna().any() or bad:
            raise ValueError(f"Store không hợp lệ: {bad[:10]} (hợp lệ: 1–{max(valid)})")
        d["Store"] = s.astype(int)

        warnings = []
        if ((d["Date"].dt.dayofweek >= 5) & (d["Promo"] == 1)).any():
            warnings.append("Dữ liệu lịch sử không có Promo vào Thứ Bảy/Chủ nhật: kết quả là ngoại suy.")
        if (d["Date"].dt.dayofweek == 6).any():
            warnings.append("Rất ít cửa hàng mở Chủ nhật trong lịch sử (≈3.600 dòng): độ tin cậy thấp.")
        if (d["StateHoliday"] != "0").any():
            warnings.append("Dữ liệu ngày lễ có doanh thu rất ít (a=694, b=145, c=71 dòng): độ tin cậy thấp.")
        if d["Date"].max() > pd.Timestamp("2015-09-17") or d["Date"].min() < pd.Timestamp("2013-01-01"):
            warnings.append("Ngày nằm ngoài khoảng dữ liệu Rossmann (2013-01-01 → 2015-09-17).")
        return d, warnings

    # ---------------------------------------------------------------- dự báo
    def predict_frame(self, df):
        d, warnings = self.validate_input(df)
        art = self.model_service.artifact
        pred = art.predict_sales(d)
        low, high = art.interval(pred)
        expected = art.expected_sales(d)

        out = d[[c for c in INPUT_COLUMNS]].copy()
        out["DayOfWeek"] = d["Date"].dt.dayofweek + 1
        out["PredictedSales"] = np.round(pred, 2)
        out["PredLow"] = np.round(low, 2)
        out["PredHigh"] = np.round(high, 2)
        out["ExpectedSales"] = np.round(expected, 2)
        # Cảnh báo: mức kỳ vọng lịch sử nằm TRÊN cận trên khoảng dự báo
        out["LowForecastAlert"] = (high < expected).astype(int)
        if "Open" in d.columns:
            out["Open"] = d["Open"].to_numpy()
        if "Sales" in d.columns:
            actual = pd.to_numeric(d["Sales"], errors="coerce").to_numpy(float)
            out["ActualSales"] = actual
            out["AbsError"] = np.round(np.abs(pred - actual), 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(actual > 0, np.abs(pred - actual) / actual, np.nan)
            out["PctError"] = np.round(pct, 4)          # NaN = N/A (Sales = 0)
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        return out, warnings

    def predict_one(self, store, date, promo, state_holiday, school_holiday):
        df = pd.DataFrame([{"Store": store, "Date": date, "Promo": promo,
                            "StateHoliday": state_holiday, "SchoolHoliday": school_holiday}])
        out, warnings = self.predict_frame(df)
        return out.iloc[0].to_dict(), warnings

    def predict_csv(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file {path}")
        df = pd.read_csv(path, dtype={"StateHoliday": str})
        return self.predict_frame(df)

    @staticmethod
    def export_csv(result, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False, encoding="utf-8-sig", na_rep="N/A")
        return path
