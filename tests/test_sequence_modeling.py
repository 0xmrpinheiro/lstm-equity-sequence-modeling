from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from Models.neural_network_forecaster import prepare_holdout
from sequence_modeling import LOOKBACK, TICKER_RE, clean_history, sample_prices, windows_for_targets


ROOT = Path(__file__).resolve().parents[1]


class SequenceTests(TestCase):
    def test_window_ends_before_target(self):
        features, labels = windows_for_targets(np.arange(9), [3, 7], lookback=3)
        np.testing.assert_array_equal(features[:, :, 0], [[0, 1, 2], [4, 5, 6]])
        np.testing.assert_array_equal(labels, [3, 7])

    def test_holdout_scaler_ignores_validation_and_test_extremes(self):
        close = sample_prices()["Close"].copy()
        cutoff = int(len(close) * 0.70)
        close.iloc[cutoff:] = 1000
        scaler, train_end, val_end, train, validation, test_indices, test = prepare_holdout(close)
        self.assertEqual(scaler.data_max_[0], close.iloc[:cutoff].max())
        self.assertEqual(train[0].shape, (train_end - LOOKBACK, LOOKBACK, 1))
        self.assertEqual(len(validation[1]), val_end - train_end)
        self.assertEqual(test_indices[0], val_end)
        self.assertEqual(len(test[1]), len(close) - val_end)

    def test_clean_history_selects_exact_ticker_and_sorts_dates(self):
        prices = pd.DataFrame(
            [[11, 20], [10, 19]], index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
            columns=pd.MultiIndex.from_tuples([("Close", "MSFT"), ("Close", "AAPL")]),
        )
        self.assertEqual(clean_history(prices, "MSFT")["Close"].tolist(), [10, 11])
        with self.assertRaises(ValueError):
            clean_history(prices, "SPY")
        self.assertIsNotNone(TICKER_RE.fullmatch("BRK-B"))
        self.assertIsNone(TICKER_RE.fullmatch("NOT A TICKER"))


class AppSmokeTests(TestCase):
    def test_offline_sample_loads_bundled_model_and_renders_both_charts(self):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=45)
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertFalse(app.error, [item.value for item in app.error])
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertTrue(any(item.value == "Historical model replay" for item in app.subheader))