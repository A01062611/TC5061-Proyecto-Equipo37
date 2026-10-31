"""Con la misma semilla, la tabla de P2 y el entrenamiento deben dar exactamente lo mismo."""

import numpy as np
import pandas as pd
import pytest

from MiBici.config import SEED
from MiBici.features import construir_dataset_p2
from MiBici.modeling.train_p2 import DEFAULT_HYPERPARAMS, train_model

K = 2

@pytest.mark.parametrize("model", list(DEFAULT_HYPERPARAMS))
def test_mismo_seed_mismas_predicciones_y_metricas(model, p2_split):
    train, val, y_train, y_val, features = p2_split
    params = DEFAULT_HYPERPARAMS[model]
    a = train_model(model, train, val, y_train, y_val, features, K, SEED, params)
    b = train_model(model, train, val, y_train, y_val, features, K, SEED, params)

    np.testing.assert_array_equal(a["proba"], b["proba"])
    np.testing.assert_array_equal(a["pred"], b["pred"])
    assert a["params"] == b["params"]
    assert a["res"] == b["res"]
    np.testing.assert_array_equal(a["estimador"].predict(val[features]), b["estimador"].predict(val[features]))
