"""ML Layer – dự báo dữ liệu mới bằng model đã lưu (không train lại).

Dòng lệnh:
    python -m src.models.predict --store 1 --date 2015-08-01 --promo 1 --state-holiday 0 --school-holiday 0
    python -m src.models.predict --csv datasets/sample/sample_input.csv --out results/predictions/forecast.csv
"""
import argparse

import pandas as pd

from src import config
from src.services.model_service import ModelService
from src.services.prediction_service import PredictionService


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dự báo doanh thu Rossmann bằng model đã lưu")
    ap.add_argument("--csv", help="File CSV có cột Store, Date, Promo, StateHoliday, SchoolHoliday")
    ap.add_argument("--out", default=str(config.OUT_PRED / "forecast.csv"))
    ap.add_argument("--store", type=int)
    ap.add_argument("--date")
    ap.add_argument("--promo", type=int, default=0)
    ap.add_argument("--state-holiday", default="0")
    ap.add_argument("--school-holiday", type=int, default=0)
    a = ap.parse_args(argv)

    service = PredictionService(ModelService())
    if a.csv:
        result, warnings = service.predict_csv(a.csv)
    elif a.store is not None and a.date:
        result, warnings = service.predict_frame(pd.DataFrame([{
            "Store": a.store, "Date": a.date, "Promo": a.promo,
            "StateHoliday": a.state_holiday, "SchoolHoliday": a.school_holiday}]))
    else:
        ap.error("Cần --csv hoặc (--store và --date)")
    for w in warnings:
        print("Cảnh báo:", w)
    print(result.to_string(index=False))
    print("Đã xuất:", service.export_csv(result, a.out))
    return result


if __name__ == "__main__":
    main()
