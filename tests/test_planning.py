import pandas as pd
import pytest

from src.services.planning_service import PlanningService, allocate


@pytest.fixture(scope="module")
def planning(prediction_service):
    return PlanningService(prediction_service)


def test_allocate_respects_budget_and_need():
    alloc = allocate([100, 300, 0], [5, 5, 0], 6)
    assert alloc.sum() == 6 and (alloc <= [5, 5, 0]).all() and alloc[1] > alloc[0]
    assert allocate([100, 300], [2, 3], 10).tolist() == [2, 3]       # đủ ca -> giữ nguyên nhu cầu


def test_staffing_formula():
    fc = pd.DataFrame({"Store": [1, 1, 1], "Date": pd.to_datetime(["2015-08-03", "2015-08-04", "2015-08-05"]),
                       "Open": [1, 1, 0], "PredictedSales": [2500.0, 500.0, 0.0], "PredHigh": [3100.0, 700.0, 0.0]})
    plan = PlanningService.staffing(fc, revenue_per_shift=1000, min_shifts=2)
    assert plan["ShiftsNeeded"].tolist() == [3, 2, 0]          # ceil(2500/1000)=3; tối thiểu 2; đóng cửa 0
    plan = PlanningService.staffing(fc, revenue_per_shift=1000, min_shifts=2, basis="PredHigh")
    assert plan["ShiftsNeeded"].tolist() == [4, 2, 0]
    with pytest.raises(ValueError):
        PlanningService.staffing(fc, revenue_per_shift=0)


def test_run_full_plan(planning):
    r = planning.run([1, 262], "2015-08-01", "2015-09-17", 1000, granularity="Tuần")
    plan = r["plan"]
    assert len(plan) == 2 * 48 and plan["Store"].nunique() == 2
    assert not plan["PredictedSales"].isna().any() and (plan["PredictedSales"] >= 0).all()
    assert (plan.loc[plan["Open"] == 0, ["PredictedSales", "ShiftsNeeded"]] == 0).all().all()
    assert r["ranking"]["Rank"].tolist() == [1, 2]
    assert r["summary"]["PredictedSales"].sum() == pytest.approx(plan["PredictedSales"].sum())
    assert set(r["suggestions"].columns) == {"Store", "Ngày", "Gợi ý", "Căn cứ"}


@pytest.mark.parametrize("granularity,expected", [("Ngày", 48), ("Tuần", 8), ("Tháng", 2)])
def test_aggregate_periods(planning, granularity, expected):
    r = planning.run([1], "2015-08-01", "2015-09-17", 1000, granularity=granularity)
    assert r["summary"]["Period"].nunique() == expected


def test_budget_allocation_and_shortage_suggestion(planning):
    r = planning.run([1, 262], "2015-08-03", "2015-08-09", 1000, shift_budget=10)
    plan = r["plan"]
    daily = plan.groupby("Date")["ShiftsAllocated"].sum()
    assert (daily <= 10).all()
    assert (r["suggestions"]["Gợi ý"] == "Thiếu ca so với nhu cầu").any()


@pytest.mark.parametrize("stores,start,end", [([], "2015-08-01", "2015-08-10"), ([817], "2015-08-01", "2015-08-10"),
                                              ([1], "2015-08-10", "2015-08-01"), ([1], "2016-01-01", "2016-02-01")])
def test_invalid_inputs(planning, stores, start, end):
    with pytest.raises(ValueError):
        planning.run(stores, start, end, 1000)


def test_planning_never_trains(planning, monkeypatch):
    art = planning.prediction_service.model_service.artifact

    def forbidden(*args, **kwargs):
        raise AssertionError("Kế hoạch không được train lại model")
    monkeypatch.setattr(art.model, "fit", forbidden)
    monkeypatch.setattr(art.pipeline, "fit", forbidden)
    planning.run([1], "2015-08-01", "2015-08-14", 1000)


def test_promo_suggestion_is_marked_as_simulation(planning):
    s = planning.run([1], "2015-08-01", "2015-09-17", 1000)["suggestions"]
    promo = s[s["Gợi ý"] == "Cân nhắc khuyến mãi"]
    assert promo.empty or promo["Căn cứ"].str.contains("mô phỏng").all()
