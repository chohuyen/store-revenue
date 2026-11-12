"""Feature Layer – RossmannPipeline: tiền xử lý + tạo đặc trưng.

Dùng CHUNG cho train, evaluate, predict và What-if để đảm bảo dữ liệu mới được xử lý
giống hệt dữ liệu huấn luyện (tách từ ô 14–15 của notebook gốc).

    pipe = RossmannPipeline().fit(train_clean, store_raw)   # chỉ fit trên tập Train
    X = pipe.transform(df)   # df chỉ cần: Store, Date, Promo, StateHoliday, SchoolHoliday
    pipe.save(path) / RossmannPipeline.load(path)

Bộ đặc trưng:
    "base"         (mô hình chính thức) – 24 đặc trưng: thông tin cửa hàng (store.csv),
                   lịch (năm, tháng, ngày, tuần, thứ, đầu/cuối tháng), Promo, SchoolHoliday,
                   cờ ngày lễ và 2 đặc trưng tương tác Promo × ngày lễ / nghỉ học.
    "holiday_type" (thí nghiệm) – base + StateHoliday mã hoá 0/a/b/c. Không tốt hơn trên
                   Validation nên không dùng cho mô hình chính (xem scripts/experiments.py).
Không đặc trưng nào dùng Sales hay Customers (Customers không biết trước khi dự báo).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.data.loader import check_schema, normalize_state_holiday

# Cột người dùng phải cung cấp khi dự báo (mọi thứ khác lấy từ store.csv hoặc từ Date)
INPUT_COLUMNS = ["Store", "Date", "Promo", "StateHoliday", "SchoolHoliday"]

BASE_FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "CompetitionOpenDateKnown",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "Year", "Month", "Day", "WeekOfYear",
    "IsWeekend", "IsMonthStart", "IsMonthEnd", "IsStateHoliday",
    "Promo_x_SchoolHoliday", "Promo_x_StateHoliday",
    "StoreType", "Assortment", "PromoInterval",
]
FEATURE_SETS = {"base": BASE_FEATURES, "holiday_type": BASE_FEATURES + ["StateHoliday"]}
CATEGORICAL = {"base": ["StoreType", "Assortment", "PromoInterval"],
               "holiday_type": ["StoreType", "Assortment", "PromoInterval", "StateHoliday"]}


class RossmannPipeline:
    def __init__(self, feature_set="base"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set phải là {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.features = FEATURE_SETS[feature_set]
        self.categorical = CATEGORICAL[feature_set]
        self.median_competition_distance = None
        self.store_table = None          # store.csv đã xử lý, lưu kèm pipeline
        self.encoder = None
        self.train_date_range = None

    # ------------------------------------------------------------ 1. store.csv
    def _preprocess_store(self, store_raw):
        s = store_raw.copy()
        s["CompetitionDistance"] = s["CompetitionDistance"].fillna(self.median_competition_distance)
        # -1 = không rõ ngày đối thủ mở cửa
        s["CompetitionOpenSinceMonth"] = s["CompetitionOpenSinceMonth"].fillna(-1).astype(int)
        s["CompetitionOpenSinceYear"] = s["CompetitionOpenSinceYear"].fillna(-1).astype(int)
        # 0 = cửa hàng không tham gia Promo2 (544 store, đúng bằng số dòng thiếu)
        s["Promo2SinceWeek"] = s["Promo2SinceWeek"].fillna(0).astype(int)
        s["Promo2SinceYear"] = s["Promo2SinceYear"].fillna(0).astype(int)
        s["PromoInterval"] = s["PromoInterval"].fillna("None")
        # 1 = biết ngày đối thủ mở cửa (notebook gốc gọi là HasCompetition, đổi tên cho đúng nghĩa)
        s["CompetitionOpenDateKnown"] = (s["CompetitionOpenSinceYear"] != -1).astype(int)
        return s

    # ------------------------------------------------------------ 2. ghép + đặc trưng
    def prepare(self, df):
        """Kiểm tra đầu vào, ghép store, tạo đặc trưng. Trả DataFrame đầy đủ (chưa mã hoá)."""
        if self.store_table is None:
            raise RuntimeError("Pipeline chưa fit()")
        check_schema(df, INPUT_COLUMNS, "Dữ liệu đầu vào")
        data = df.copy()
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        if data["Date"].isna().any():
            raise ValueError("Có giá trị Date không đọc được (định dạng đúng: YYYY-MM-DD)")
        data["StateHoliday"] = normalize_state_holiday(data["StateHoliday"])
        for col in ["Promo", "SchoolHoliday"]:
            vals = pd.to_numeric(data[col], errors="coerce")
            if vals.isna().any() or not vals.isin([0, 1]).all():
                raise ValueError(f"{col} chỉ nhận 0 hoặc 1")
            data[col] = vals.astype(int)
        data["Store"] = pd.to_numeric(data["Store"], errors="coerce")
        unknown = sorted(set(data["Store"].dropna().astype(int)) - set(self.store_table["Store"]))
        if data["Store"].isna().any() or unknown:
            raise ValueError(f"Store không tồn tại trong store.csv: {unknown[:10]}")
        data["Store"] = data["Store"].astype(int)

        # Bỏ các cột store có sẵn trong input (nếu có) để luôn lấy từ store.csv
        store_cols = [c for c in self.store_table.columns if c != "Store"]
        data = data.drop(columns=[c for c in store_cols if c in data.columns])
        data = data.merge(self.store_table, on="Store", how="left")

        # Đặc trưng thời gian (DayOfWeek tính lại từ Date: 1 = Thứ Hai ... 7 = Chủ nhật)
        data["DayOfWeek"] = data["Date"].dt.dayofweek + 1
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
        data["IsWeekend"] = (data["DayOfWeek"] >= 6).astype(int)
        data["IsMonthStart"] = data["Date"].dt.is_month_start.astype(int)
        data["IsMonthEnd"] = data["Date"].dt.is_month_end.astype(int)
        # Ngày lễ và tương tác (giữ như notebook)
        data["IsStateHoliday"] = (data["StateHoliday"] != "0").astype(int)
        data["Promo_x_SchoolHoliday"] = data["Promo"] * data["SchoolHoliday"]
        data["Promo_x_StateHoliday"] = data["Promo"] * data["IsStateHoliday"]
        return data

    # ------------------------------------------------------------ 3. fit / transform
    def fit(self, train_clean, store_raw):
        """Học median CompetitionDistance (từ store.csv) và OrdinalEncoder (từ Train)."""
        self.median_competition_distance = float(store_raw["CompetitionDistance"].median())
        self.store_table = self._preprocess_store(store_raw)
        prepared = self.prepare(train_clean)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.encoder.fit(prepared[self.categorical].astype(str))
        self.train_date_range = [str(prepared["Date"].min().date()), str(prepared["Date"].max().date())]
        return self

    def encode(self, prepared):
        """Từ DataFrame của prepare() -> ma trận đặc trưng X (đúng thứ tự self.features)."""
        X = prepared[self.features].copy()
        X[self.categorical] = self.encoder.transform(X[self.categorical].astype(str))
        X = X.astype(float)
        if X.isna().any().any():
            raise ValueError(f"Đặc trưng còn NaN: {X.columns[X.isna().any()].tolist()}")
        return X

    def transform(self, df):
        return self.encode(self.prepare(df))

    # ------------------------------------------------------------ 4. lưu / nạp
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        obj = joblib.load(path)
        if not isinstance(obj, RossmannPipeline):
            raise TypeError(f"{path} không phải RossmannPipeline")
        return obj

    def store_ids(self):
        return self.store_table["Store"].astype(int).tolist()


def target_log(sales):
    """Target dùng khi train: log1p(Sales) – giảm lệch phải (skew 1.59 -> -0.11)."""
    return np.log1p(np.asarray(sales, dtype=float))


def inverse_target(pred_log):
    """Đổi dự báo từ thang log về €: expm1, rồi kẹp >= 0 vì doanh thu không âm."""
    return np.maximum(np.expm1(np.asarray(pred_log, dtype=float)), 0.0)
