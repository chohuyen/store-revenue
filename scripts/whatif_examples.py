"""Sinh ví dụ What-if bằng MODEL ĐÃ LƯU cho báo cáo (thay số liệu của 'mô hình thay thế' trong DRAFT).
Chạy: python scripts/whatif_examples.py -> results/evaluation/whatif_examples.csv"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.services.model_service import ModelService  # noqa: E402
from src.services.prediction_service import PredictionService  # noqa: E402
from src.services.whatif_service import WhatIfService  # noqa: E402

ws = WhatIfService(PredictionService(ModelService()))
NO = {"Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0}
CASES = [("Bật khuyến mãi", NO, {"Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0}),
         ("Ngày lễ công (a)", NO, {"Promo": 0, "StateHoliday": "a", "SchoolHoliday": 0}),
         ("Nghỉ học", NO, {"Promo": 0, "StateHoliday": "0", "SchoolHoliday": 1})]
rows = []
for store in [1, 262, 292, 817, 1114]:
    for name, base, scen in CASES:
        r = ws.compare(store, "2015-08-05", base, scen)        # Thứ Tư
        rows.append({"Store": store, "Date": "2015-08-05", "Scenario": name,
                     "BaselineSales": round(r["baseline_sales"], 2), "ScenarioSales": round(r["scenario_sales"], 2),
                     "AbsDiff": round(r["abs_diff"], 2),
                     "PctDiff": None if r["pct_diff"] is None else round(r["pct_diff"], 2)})
df = pd.DataFrame(rows)
out = config.OUT_EVAL / "whatif_examples.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print(df.groupby("Scenario")["PctDiff"].describe().round(2).to_string())
