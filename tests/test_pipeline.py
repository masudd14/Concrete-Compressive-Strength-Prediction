"""Unit tests for Concrete ML pipeline components."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import ConcreteFeatureEngineering
from src.evaluate import evaluate_regression
from src.data_loader import CLEAN_FEATURE_NAMES, RAW_COLUMNS, TARGET_NAME


class TestConcreteML(unittest.TestCase):
    """Test suite for domain features and evaluation."""

    def test_schema_lengths(self):
        """Ensure feature lists match UCI concrete data specification."""
        self.assertEqual(len(CLEAN_FEATURE_NAMES), 8)
        self.assertEqual(len(RAW_COLUMNS), 9)
        self.assertEqual(TARGET_NAME, "compressive_strength")

    def test_feature_engineering_math(self):
        """Validate engineering calculation against Abrams law and binder definitions."""
        fe = ConcreteFeatureEngineering()
        # [cement, slag, fly_ash, water, sp, coarse, fine, age]
        sample = np.array([[300.0, 100.0, 50.0, 180.0, 10.0, 950.0, 750.0, 28.0]])
        out = fe.transform(sample)

        # 8 original + 6 engineered = 14 total columns
        self.assertEqual(out.shape[1], 14)

        # Check total binder: cement (300) + slag (100) + fly_ash (50) = 450
        expected_binder = 450.0
        self.assertAlmostEqual(out[0, 8], expected_binder)

        # Check water / cement ratio: 180 / 300 = 0.6
        self.assertAlmostEqual(out[0, 9], 180.0 / 300.0, places=4)

        # Check water / binder ratio: 180 / 450 = 0.4
        self.assertAlmostEqual(out[0, 10], 180.0 / 450.0, places=4)

        # Check log_age: log1p(28)
        self.assertAlmostEqual(out[0, 13], np.log1p(28.0), places=4)

    def test_evaluation_metrics(self):
        """Verify regression metric calculation on synthetic arrays."""
        y_true = np.array([30.0, 40.0, 50.0])
        y_pred = np.array([30.0, 40.0, 50.0])
        metrics = evaluate_regression(y_true, y_pred)

        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["r2_score"], 1.0)
        self.assertAlmostEqual(metrics["rmse_percentage_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
