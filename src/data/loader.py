"""Data Layer: đọc, kiểm tra và làm sạch dữ liệu Rossmann.

Các hàm chính:
    load_train(), load_test(), load_store()  -> đọc CSV với kiểu dữ liệu đúng
    check_schema(df, required)               -> báo lỗi nếu thiếu cột
    data_quality_report()                    -> dict thống kê missing/duplicate/kiểu dữ liệu
    clean_train(train)                       -> bỏ ngày đóng cửa (Open=0) và Sales<=0
    split_by_time(df)                        -> (train, val, test) theo mốc trong config

Chạy thử:  python -m src.data.loader
"""
import json
import zipfile

from pathlib import Path

import pandas as pd

from src import config

TRAIN_COLUMNS = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open",
                 "Promo", "StateHoliday", "SchoolHoliday"]
TEST_COLUMNS = ["Id", "Store", "DayOfWeek", "Date", "Open", "Promo",
                "StateHoliday", "SchoolHoliday"]
STORE_COLUMNS = ["Store", "StoreType", "Assortment", "CompetitionDistance",
                 "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
                 "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]

# StateHoliday trong train.csv trộn số 0 và chuỗi "0" -> luôn đọc dạng chuỗi
_DTYPES = {"StateHoliday": str}


def check_schema(df, required, name="dữ liệu"):
    """Kiểm tra df có đủ cột cần thiết; thiếu thì raise ValueError."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} thiếu cột: {missing}")
    return True


def normalize_state_holiday(series):
    """Đưa StateHoliday về 4 giá trị '0','a','b','c' (0, 0.0, '0' -> '0')."""
    s = series.astype(str).str.strip().replace({"0.0": "0", "nan": "0", "": "0"})
    bad = sorted(set(s) - set(config.STATE_HOLIDAY_VALUES))
    if bad:
        raise ValueError(f"StateHoliday không hợp lệ: {bad}. Chỉ nhận {config.STATE_HOLIDAY_VALUES}")
    return s


def _read(path, columns, name):
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path.name}. Hãy chép {path.name} (hoặc file "
                                f"rossmann-store-sales.zip của Kaggle) vào thư mục datasets/")
    df = pd.read_csv(path, dtype=_DTYPES, low_memory=False)
    check_schema(df, columns, name)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if "StateHoliday" in df.columns:
        df["StateHoliday"] = normalize_state_holiday(df["StateHoliday"])
    return df


def ensure_train_csv(path=config.TRAIN_CSV):
    """Nếu chưa có train.csv nhưng trong data/raw có file zip của Kaggle (rossmann-store-sales.zip)
    thì tự giải nén train.csv ra. Trả về True nếu sau đó có train.csv."""
    if path.exists():
        return True
    for archive in sorted(path.parent.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            member = next((n for n in zf.namelist() if Path(n).name == "train.csv"), None)
            if member:
                path.write_bytes(zf.read(member))
                return True
    return False


def load_train(path=config.TRAIN_CSV):
    ensure_train_csv(path)
    return _read(path, TRAIN_COLUMNS, "train.csv")


def load_test(path=config.TEST_CSV):
    df = _read(path, TEST_COLUMNS, "test.csv")
    # 11 dòng Open bị thiếu (Store 622): coi là mở cửa, giống notebook gốc
    df["Open"] = df["Open"].fillna(1).astype(int)
    return df


def load_store(path=config.STORE_CSV):
    df = _read(path, STORE_COLUMNS, "store.csv")
    if df["Store"].duplicated().any():
        raise ValueError("store.csv có Store bị trùng")
    return df


def clean_train(train):
    """Quy tắc làm sạch của dự án (không đổi so với notebook):
    giữ lại ngày mở cửa và có doanh thu: Open != 0 và Sales > 0."""
    return train[(train["Open"] != 0) & (train["Sales"] > 0)].copy()


def split_by_time(df):
    """Chia theo thời gian, không xáo trộn -> tránh dùng dữ liệu tương lai để train."""
    d = df["Date"]
    train = df[d <= config.TRAIN_END].copy()
    val = df[(d >= config.VAL_START) & (d <= config.VAL_END)].copy()
    test = df[(d >= config.TEST_START) & (d <= config.TEST_END)].copy()
    return train, val, test


def _summary(df):
    return {
        "rows": int(len(df)),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def data_quality_report():
    """Thống kê chất lượng dữ liệu cho cả 3 file + kết quả làm sạch và chia tập."""
    train, test, store = load_train(), load_test(), load_store()
    clean = clean_train(train)
    tr, va, te = split_by_time(clean)
    return {
        "train.csv": {**_summary(train),
                      "duplicate_store_date": int(train.duplicated(["Store", "Date"]).sum()),
                      "date_range": [str(train.Date.min().date()), str(train.Date.max().date())],
                      "open0_rows": int((train.Open == 0).sum()),
                      "open1_sales0_rows": int(((train.Open == 1) & (train.Sales == 0)).sum())},
        "test.csv": {**_summary(test),
                     "date_range": [str(test.Date.min().date()), str(test.Date.max().date())],
                     "note": "Không có Sales/Customers; Open thiếu được điền 1"},
        "store.csv": _summary(store),
        "stores_in_test_not_in_store": int(len(set(test.Store) - set(store.Store))),
        "clean_rows": int(len(clean)),
        "split_rows": {"train": len(tr), "val": len(va), "test": len(te)},
    }


if __name__ == "__main__":
    report = data_quality_report()
    config.OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out = config.OUT_EVAL / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("Đã lưu:", out)
