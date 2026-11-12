"""ModelService: nạp model một lần, cung cấp metadata và bảng so sánh mô hình cho UI."""
import pandas as pd

from src import config
from src.models.model_loader import METADATA_PATH, MODEL_PATH, load_artifact


class ModelService:
    def __init__(self, model_path=MODEL_PATH, meta_path=METADATA_PATH):
        self.model_path, self.meta_path = model_path, meta_path
        self._artifact = None

    def load(self, force=False):
        """Nạp model từ đĩa (cache lại, không train)."""
        if self._artifact is None or force:
            self._artifact = load_artifact(self.model_path, self.meta_path)
        return self._artifact

    @property
    def artifact(self):
        return self.load()

    def is_available(self):
        return self.model_path.exists() and self.meta_path.exists()

    def metadata(self):
        return self.artifact.metadata

    def store_ids(self):
        return self.artifact.pipeline.store_ids()

    def store_info(self, store):
        t = self.artifact.pipeline.store_table
        row = t[t["Store"] == int(store)]
        if row.empty:
            raise ValueError(f"Store {store} không tồn tại")
        cols = ["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2", "PromoInterval"]
        return row[cols].iloc[0].to_dict()

    @staticmethod
    def comparison_table(path=config.OUT_EVAL / "model_comparison.csv"):
        if not path.exists():
            raise FileNotFoundError("Chưa có results/evaluation/model_comparison.csv – hãy chạy train")
        return pd.read_csv(path, encoding="utf-8-sig")

    def retrain(self):
        """Train lại toàn bộ (chậm, vài phút). UI phải gọi trong QThread."""
        from src.models import train
        train.main()
        return self.load(force=True)
