import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


from src.data.loader import ensure_train_csv  # noqa: E402

HAS_TRAIN = ensure_train_csv()          # giải nén datasets/train.zip nếu cần
needs_train = pytest.mark.skipif(not HAS_TRAIN, reason="Không có datasets/train.csv hoặc train.zip")


@pytest.fixture(scope="session")
def model_service():
    from src.services.model_service import ModelService
    service = ModelService()
    service.load()
    return service


@pytest.fixture(scope="session")
def prediction_service(model_service):
    from src.services.prediction_service import PredictionService
    return PredictionService(model_service)


@pytest.fixture(scope="session")
def whatif_service(prediction_service):
    from src.services.whatif_service import WhatIfService
    return WhatIfService(prediction_service)


@pytest.fixture
def one_row():
    return pd.DataFrame([{"Store": 1, "Date": "2015-08-03", "Promo": 1, "StateHoliday": "0", "SchoolHoliday": 0}])
