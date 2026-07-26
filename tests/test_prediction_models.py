import unittest
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class _ScoreModelStub:
    def predict(self, data):
        rows = len(data)
        return np.full(rows, 4.0), np.full(rows, 3.0)


class _CalibratorStub:
    def predict(self, probability):
        return np.asarray(probability, dtype=float)


class PredictionModelTests(unittest.TestCase):
    def test_score_model_exposes_binary_predict_proba_contract(self):
        from prediction_models import CalibratedScoreProbabilityModel

        model = CalibratedScoreProbabilityModel(
            score_model=_ScoreModelStub(),
            calibrator=_CalibratorStub(),
            feature_columns=["feature"],
        )
        probability = model.predict_proba(
            pd.DataFrame({"feature": [1.0, 2.0]})
        )

        self.assertEqual(probability.shape, (2, 2))
        np.testing.assert_allclose(probability.sum(axis=1), 1.0)
        self.assertTrue((probability[:, 1] > 0.5).all())

    def test_joblib_roundtrip_preserves_predictions(self):
        from prediction_models import CalibratedScoreProbabilityModel

        model = CalibratedScoreProbabilityModel(
            score_model=_ScoreModelStub(),
            calibrator=_CalibratorStub(),
            feature_columns=["feature"],
        )
        data = pd.DataFrame({"feature": [1.0, 2.0]})
        original = model.predict_proba(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            restored = loaded.predict_proba(data)

        np.testing.assert_allclose(original, restored)
        self.assertEqual(loaded.feature_columns, ["feature"])


if __name__ == "__main__":
    unittest.main()
