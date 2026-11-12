import pandas as pd
import pytest

from src.data.loader import (STORE_COLUMNS, check_schema, clean_train, load_store, load_test,
                             normalize_state_holiday, split_by_time)
from tests.conftest import needs_train


def test_load_store_schema_and_unique():
    store = load_store()
    assert list(store.columns) == STORE_COLUMNS
    assert len(store) == 1115 and store.Store.is_unique


def test_load_test_has_no_sales_or_customers_and_open_filled():
    test = load_test()
    assert "Sales" not in test.columns and "Customers" not in test.columns
    assert test.Open.isna().sum() == 0


def test_check_schema_raises_on_missing_column():
    with pytest.raises(ValueError, match="thiếu cột"):
        check_schema(pd.DataFrame({"Store": [1]}), ["Store", "Date"])


def test_normalize_state_holiday_mixed_types():
    s = normalize_state_holiday(pd.Series([0, "0", "a", "b", "c", 0.0]))
    assert s.tolist() == ["0", "0", "a", "b", "c", "0"]


def test_normalize_state_holiday_rejects_invalid():
    with pytest.raises(ValueError):
        normalize_state_holiday(pd.Series(["x"]))


def test_clean_train_rule():
    df = pd.DataFrame({"Open": [0, 1, 1, 1], "Sales": [0, 0, 10, 5]})
    assert clean_train(df)["Sales"].tolist() == [10, 5]


def test_split_by_time_boundaries():
    dates = pd.to_datetime(["2015-05-07", "2015-05-08", "2015-06-18", "2015-06-19", "2015-07-31"])
    tr, va, te = split_by_time(pd.DataFrame({"Date": dates}))
    assert len(tr) == 1 and len(va) == 2 and len(te) == 2
    assert tr.Date.max() < va.Date.min() and va.Date.max() < te.Date.min()


@needs_train
def test_real_split_sizes():
    from src.data.loader import load_train
    tr, va, te = split_by_time(clean_train(load_train()))
    assert (len(tr), len(va), len(te)) == (765470, 37472, 41396)


def test_ensure_train_csv_extracts_from_kaggle_zip(tmp_path):
    import zipfile

    from src.data.loader import ensure_train_csv
    with zipfile.ZipFile(tmp_path / "rossmann-store-sales.zip", "w") as zf:
        zf.writestr("train.csv", "Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday\n")
    target = tmp_path / "train.csv"
    assert ensure_train_csv(target) and target.exists()


def test_missing_train_message(tmp_path):
    from src.data.loader import load_train
    with pytest.raises(FileNotFoundError, match="rossmann-store-sales.zip"):
        load_train(tmp_path / "train.csv")
