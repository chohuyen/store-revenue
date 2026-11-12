import pandas as pd
import pytest

from src.services.dashboard_service import MONTHLY_COLUMNS, DashboardService, build_monthly


@pytest.fixture
def daily():
    return pd.DataFrame({
        "Store": [1, 1, 2, 2, 3], "Sales": [100.0, 200.0, 50.0, 70.0, 500.0], "Promo": [0, 1, 0, 1, 0],
        "Date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-01", "2015-02-01", "2015-02-02"]),
        "StoreType": ["a", "a", "b", "b", "a"], "Assortment": ["a"] * 5})


@pytest.fixture
def service(daily):
    return DashboardService(daily=daily)


def test_kpis_from_real_rows(service):
    k = service.kpis()
    assert k["total_sales"] == 920 and k["stores"] == 3 and k["rows"] == 5 and k["months"] == 2
    assert k["avg_sales_per_store_day"] == pytest.approx(184)
    assert k["avg_sales_promo"] == pytest.approx(135) and k["avg_sales_no_promo"] == pytest.approx(650 / 3)


def test_filters_and_top_stores(service):
    top = service.top_stores(n=2, store_type="a")
    assert top["Store"].tolist() == [3, 1]
    assert top["AvgDailySales"].tolist() == [500, 150]
    assert service.kpis(start="2015-02-10")["rows"] == 2          # lọc theo tháng


def test_empty_range_raises(service):
    with pytest.raises(ValueError):
        service.kpis(start="2020-01-01")


def test_compare_stores(service):
    t = service.compare_stores([1, 2])
    assert list(t.columns) == [1, 2] and t.loc[pd.Timestamp("2015-01-01"), 1] == 300


def test_monthly_table_columns_and_totals(daily):
    m = build_monthly(daily)
    assert list(m.columns) == MONTHLY_COLUMNS
    assert m["Sales"].sum() == daily["Sales"].sum() and m["Rows"].sum() == len(daily)


def test_store_history(service):
    h = service.store_history(1, "2015-02-01")
    assert h["Sales"].tolist() == [100.0, 200.0]
