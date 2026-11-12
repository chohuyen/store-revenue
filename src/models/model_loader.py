"""ML Layer – nạp model đã lưu.

File được tạo bởi `python -m src.models.train`:
    trained/sales_model.joblib   : {"model", "pipeline", "baseline_a"}
    trained/model_metadata.json  : tên model, feature, tham số, metric, thời điểm train, version

    art = load_artifact()
    sales = art.predict_sales(df)   # df: Store, Date, Promo, StateHoliday, SchoolHoliday (+ Open tuỳ chọn)
"""
import json
from pathlib import Path

import joblib
import numpy as np

from src import config
from src.features.pipeline import inverse_target

MODEL_PATH = config.MODELS_DIR / "sales_model.joblib"
METADATA_PATH = config.MODELS_DIR / "model_metadata.json"


class SalesModelArtifact:
    def __init__(self, model, pipeline, baseline_a, metadata):
        self.model = model
        self.pipeline = pipeline
        self.baseline_a = baseline_a
        self.metadata = metadata

    @property
    def name(self):
        return self.metadata["model_name"]

    def predict_sales(self, df):
        """Dự báo doanh thu (€). Không gọi fit() – chỉ transform + predict.
        Dòng có Open = 0 (nếu có cột Open) trả về 0 vì cửa hàng đóng cửa."""
        prepared = self.pipeline.prepare(df)
        X = self.pipeline.encode(prepared)
        pred = inverse_target(self.model.predict(X))
        if "Open" in df.columns:
            pred = np.where(df["Open"].to_numpy() == 0, 0.0, pred)
        return pred

    def interval(self, pred):
        """Khoảng dự báo min–max thực nghiệm (xem evaluate.fit_interval)."""
        iv = self.metadata["prediction_interval"]
        pred = np.asarray(pred, float)
        return pred * iv["ratio_low"], pred * iv["ratio_high"]

    def expected_sales(self, df):
        """Mức 'kỳ vọng' theo lịch sử = Baseline A (trung bình Store-DayOfWeek-Promo trên Train)."""
        prepared = self.pipeline.prepare(df)
        return self.baseline_a.predict(prepared)


def save_artifact(model, pipeline, baseline_a, metadata, model_path=MODEL_PATH, meta_path=METADATA_PATH):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "pipeline": pipeline, "baseline_a": baseline_a}, model_path, compress=3)
    Path(meta_path).write_text(json.dumps(metadata, indent=1, ensure_ascii=False), encoding="utf-8")


def load_artifact(model_path=MODEL_PATH, meta_path=METADATA_PATH):
    model_path, meta_path = Path(model_path), Path(meta_path)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Chưa có model ({model_path.name}). Hãy chạy: python -m src.models.train")
    obj = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if obj["pipeline"].features != metadata["features"]:
        raise ValueError("Danh sách feature của pipeline và metadata không khớp")
    return SalesModelArtifact(obj["model"], obj["pipeline"], obj["baseline_a"], metadata)
