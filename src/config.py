"""Cấu hình chung: đường dẫn (pathlib, tương đối với thư mục dự án), mốc chia dữ liệu, seed.

Chạy được trên Windows/macOS/Linux mà không cần sửa đường dẫn.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ROOT / "datasets"                 # dữ liệu Rossmann (giống thư mục datasets/ trong bài thực hành)
DATA_SAMPLE = DATASETS / "sample"            # CSV mẫu để demo dự báo
MODELS_DIR = ROOT / "trained"                # model đã train (giống thư mục trained/ trong bài thực hành)
RESULTS = ROOT / "results"
OUT_EVAL = RESULTS / "evaluation"            # bảng metric, sai số, SHAP, thí nghiệm (CSV/JSON)
OUT_FIG = RESULTS / "figures"                # hình EDA, hình kết quả mô hình, ảnh giao diện
OUT_PRED = RESULTS / "predictions"           # file dự báo xuất ra

TRAIN_CSV = DATASETS / "train.csv"           # tự giải nén từ datasets/train.zip khi cần
TEST_CSV = DATASETS / "test.csv"
STORE_CSV = DATASETS / "store.csv"

# Chia theo thời gian
TRAIN_END = "2015-05-07"
VAL_START, VAL_END = "2015-05-08", "2015-06-18"
TEST_START, TEST_END = "2015-06-19", "2015-07-31"

SEED = 42
STATE_HOLIDAY_VALUES = ["0", "a", "b", "c"]   # 0 = không lễ, a = lễ công, b = Phục sinh, c = Giáng sinh
