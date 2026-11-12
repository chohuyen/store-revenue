import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import FEATURE_SETS, inverse_target, target_log


def test_features_match_metadata(model_service):
    art = model_service.artifact
    assert art.pipeline.features == art.metadata["features"]
    assert "Customers" not in art.pipeline.features and "Sales" not in art.pipeline.features


def test_no_leaky_columns_in_any_feature_set():
    for feats in FEATURE_SETS.values():
        assert not {"Sales", "Customers", "LogSales"} & set(feats)


def test_transform_consistent_and_no_nan(model_service, one_row):
    pipe = model_service.artifact.pipeline
    X1, X2 = pipe.transform(one_row), pipe.transform(one_row)
    assert list(X1.columns) == pipe.features
    assert not X1.isna().any().any()
    pd.testing.assert_frame_equal(X1, X2)


def test_dayofweek_derived_from_date(model_service, one_row):
    X = model_service.artifact.pipeline.transform(one_row)   # 2015-08-03 là Thứ Hai
    assert X["DayOfWeek"].iloc[0] == 1


def test_store_attributes_taken_from_store_csv(model_service, one_row):
    prepared = model_service.artifact.pipeline.prepare(one_row.assign(StoreType="z"))
    assert prepared["StoreType"].iloc[0] == "c"            # Store 1 là StoreType c


@pytest.mark.parametrize("col,value", [("Store", 99999), ("Promo", 2), ("SchoolHoliday", -1),
                                       ("StateHoliday", "x"), ("Date", "not-a-date")])
def test_invalid_inputs_raise(model_service, one_row, col, value):
    bad = one_row.copy()
    bad[col] = value
    with pytest.raises(ValueError):
        model_service.artifact.pipeline.transform(bad)


def test_log_target_round_trip_and_non_negative():
    sales = np.array([0.0, 1.0, 5000.0, 41551.0])
    assert np.allclose(inverse_target(target_log(sales)), sales)
    assert (inverse_target(np.array([-5.0, -1.0])) >= 0).all()


def test_shipped_model_uses_official_feature_set(model_service):
    from src.features.pipeline import BASE_FEATURES
    meta = model_service.metadata()
    assert meta["feature_set"] == "base" and meta["features"] == BASE_FEATURES
    assert "HasCompetition" not in meta["features"]


def test_pipeline_fit_only_on_train_rows(model_service):
    # Khoảng ngày pipeline đã học phải kết thúc trước Validation (không rò rỉ dữ liệu tương lai)
    from src import config
    assert model_service.artifact.pipeline.train_date_range[1] <= config.TRAIN_END
