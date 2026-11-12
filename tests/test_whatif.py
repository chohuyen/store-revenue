import pytest

from src.services.whatif_service import DISCLAIMER, percent_change

BASE = {"Promo": 0, "StateHoliday": "0", "SchoolHoliday": 0}
PROMO = {"Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0}


def test_percent_change_zero_base_is_none():
    assert percent_change(0, 100) is None
    assert percent_change(100, 150) == pytest.approx(50.0)


def test_compare_promo_scenario(whatif_service):
    r = whatif_service.compare(1, "2015-08-05", BASE, PROMO)
    assert len(r["table"]) == 2
    assert r["abs_diff"] == pytest.approx(r["scenario_sales"] - r["baseline_sales"])
    assert r["pct_diff"] == pytest.approx(r["abs_diff"] / r["baseline_sales"] * 100)
    assert r["disclaimer"] == DISCLAIMER and "nhân quả" in DISCLAIMER


def test_same_scenario_zero_difference(whatif_service):
    r = whatif_service.compare(262, "2015-08-05", BASE, BASE)
    assert r["abs_diff"] == 0 and r["pct_diff"] == 0


def test_other_inputs_held_constant(whatif_service):
    t = whatif_service.compare(5, "2015-08-05", BASE, PROMO)["table"]
    assert t["Store"].nunique() == 1 and t["Date"].nunique() == 1 and t["SchoolHoliday"].nunique() == 1


def test_grid_has_all_combinations(whatif_service):
    grid, _ = whatif_service.scenario_grid(1, "2015-08-05", 0)
    assert len(grid) == 8
    assert set(zip(grid.Promo, grid.StateHoliday, strict=True)) == {(p, h) for p in (0, 1) for h in "0abc"}
    assert (grid["PredictedSales"] >= 0).all()


def test_whatif_never_calls_fit(whatif_service, monkeypatch):
    art = whatif_service.prediction_service.model_service.artifact

    def forbidden(*args, **kwargs):
        raise AssertionError("What-if không được gọi fit()")
    monkeypatch.setattr(art.model, "fit", forbidden)
    monkeypatch.setattr(art.pipeline, "fit", forbidden)
    monkeypatch.setattr(art.baseline_a, "fit", forbidden)
    whatif_service.compare(1, "2015-08-05", BASE, PROMO)
    whatif_service.scenario_grid(1, "2015-08-05", 1)


def test_invalid_store_raises(whatif_service):
    with pytest.raises(ValueError):
        whatif_service.compare(0, "2015-08-05", BASE, PROMO)
