import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.model_loader import load_artifact
from tests.conftest import needs_train


def test_load_artifact_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "x.joblib", tmp_path / "x.json")


def test_metadata_fields(model_service):
    m = model_service.metadata()
    for key in ["version", "trained_at", "model_name", "features", "hyperparameters", "metrics",
                "prediction_interval", "selection_rule"]:
        assert key in m
    assert "Test" not in m["selection_rule"].split(";")[0]   # chọn theo Validation


@needs_train
def test_saved_model_reproduces_validation_metric(model_service):
    from src.data.loader import clean_train, load_train, split_by_time
    from src.models.evaluate import evaluate_predictions
    _, va, _ = split_by_time(clean_train(load_train()))
    pred = model_service.artifact.predict_sales(va)
    assert evaluate_predictions(va.Sales, pred)["RMSPE"] == pytest.approx(
        model_service.metadata()["metrics"]["val"]["RMSPE"], abs=1e-9)


def test_predict_one_valid(prediction_service):
    row, warnings = prediction_service.predict_one(1, "2015-08-03", 1, "0", 0)
    assert np.isfinite(row["PredictedSales"]) and row["PredictedSales"] > 0
    assert row["PredLow"] <= row["PredictedSales"] <= row["PredHigh"]


@pytest.mark.parametrize("store", [1, 2, 262, 558, 1115])
@pytest.mark.parametrize("date", ["2013-01-02", "2014-12-24", "2015-08-01", "2015-08-02", "2016-03-01"])
@pytest.mark.parametrize("promo,holiday", [(0, "0"), (1, "a"), (0, "b"), (1, "c")])
def test_changing_inputs_never_breaks(prediction_service, store, date, promo, holiday):
    out, _ = prediction_service.predict_frame(pd.DataFrame([{
        "Store": store, "Date": date, "Promo": promo, "StateHoliday": holiday, "SchoolHoliday": 1}]))
    assert np.isfinite(out["PredictedSales"]).all() and (out["PredictedSales"] >= 0).all()


def test_batch_csv_with_actual(prediction_service):
    out, _ = prediction_service.predict_csv(config.DATA_SAMPLE / "sample_input_with_actual.csv")
    assert len(out) == 20
    assert {"ActualSales", "AbsError", "PctError"} <= set(out.columns)
    assert not out["PredictedSales"].isna().any()
    assert np.allclose(out["AbsError"], (out["PredictedSales"] - out["ActualSales"]).abs(), atol=0.01)


def test_kaggle_test_csv_open0_predicts_zero(prediction_service):
    df = pd.read_csv(config.TEST_CSV, dtype={"StateHoliday": str}).head(3000)
    df["Open"] = df["Open"].fillna(1)
    out, _ = prediction_service.predict_frame(df)
    closed = df["Open"].to_numpy() == 0
    assert closed.any() and (out.loc[closed, "PredictedSales"] == 0).all()
    assert not out["PredictedSales"].isna().any()


def test_pct_error_na_when_actual_zero(prediction_service, one_row):
    out, _ = prediction_service.predict_frame(one_row.assign(Sales=0))
    assert np.isnan(out["PctError"].iloc[0])


def test_missing_column_message(prediction_service):
    with pytest.raises(ValueError, match="Thiếu cột"):
        prediction_service.predict_frame(pd.DataFrame([{"Store": 1, "Date": "2015-08-03"}]))


def test_empty_input(prediction_service):
    with pytest.raises(ValueError):
        prediction_service.predict_frame(pd.DataFrame())


def test_weekend_promo_warning(prediction_service):
    _, warnings = prediction_service.predict_one(1, "2015-08-01", 1, "0", 0)   # Thứ Bảy
    assert any("ngoại suy" in w for w in warnings)


def test_export_csv_round_trip(prediction_service, one_row, tmp_path):
    out, _ = prediction_service.predict_frame(one_row.assign(Sales=0))
    path = prediction_service.export_csv(out, tmp_path / "sub" / "f.csv")
    back = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    assert back["PctError"].iloc[0] == "N/A"
    assert float(back["PredictedSales"].iloc[0]) == pytest.approx(out["PredictedSales"].iloc[0])


def test_comparison_table_has_all_models(model_service):
    t = model_service.comparison_table()
    for name in ["Baseline A", "Baseline B", "Linear Regression", "Decision Tree", "Random Forest", "XGBoost",
                 "LightGBM"]:
        assert t["Model"].str.startswith(name).any()
    assert t["Selected"].sum() == 1
    sel = t[t.Selected].iloc[0]
    # chọn trong các mô hình hồi quy dùng chung (không tính baseline và Prophet chạy riêng từng cửa hàng)
    ml = t[~t.Model.str.startswith(("Baseline", "Prophet"))]
    assert sel["Val RMSPE"] == ml["Val RMSPE"].min()         # chọn theo Validation, không theo Test
    meta = json.loads((config.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert sel["Model"] == meta["model_name"]


def test_shap_importance_shape(model_service, prediction_service):
    from src.models.evaluate import shap_importance
    df = pd.read_csv(config.DATA_SAMPLE / "sample_input_with_actual.csv", dtype={"StateHoliday": str})
    X = model_service.artifact.pipeline.transform(df)
    imp, contrib = shap_importance(model_service.artifact.model, X)
    assert list(sorted(imp.index)) == sorted(X.columns) and (imp >= 0).all()
    assert contrib.shape == X.shape


def test_metrics_include_mse_r2():
    from src.models.evaluate import evaluate_predictions
    m = evaluate_predictions([100, 200, 300], [110, 190, 300])
    assert m["MSE"] == pytest.approx(200 / 3) and m["RMSE"] == pytest.approx((200 / 3) ** 0.5)
    assert m["R2"] == pytest.approx(1 - 200 / 20000)
    assert evaluate_predictions([1, 2, 3], [1, 2, 3])["R2"] == 1


def test_linear_regression_uses_one_hot_store():
    from src.models.train import make_model
    model = make_model("Linear Regression")
    assert "Store" in model.steps[0][1].transformers[0][2]
