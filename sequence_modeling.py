"""Shared data preparation for the Streamlit demo and chronological benchmark."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


LOOKBACK = 100
MODEL_PATH = Path(__file__).resolve().parent / "Models" / "neural_forecaster.keras"
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.-][A-Z]{1,2})?$")


def load_model(path=MODEL_PATH):
    """Load the trusted, bundled Keras artifact without loading optimizer state."""
    from keras.models import load_model as keras_load_model

    return keras_load_model(path, compile=False, safe_mode=True)


def sample_prices():
    """Make a deterministic, synthetic price path for offline demonstrations."""
    rng = np.random.default_rng(42)
    steps = np.arange(420)
    prices = 100 + 0.11 * steps + 4 * np.sin(steps / 14) + rng.normal(0, 1.2, len(steps))
    return pd.DataFrame(
        {"Close": prices}, index=pd.bdate_range("2023-01-02", periods=len(steps), name="Date")
    )


def clean_history(frame, ticker=None):
    """Select the requested ticker and normalize the index and numeric columns."""
    if frame.empty:
        raise ValueError("No observations were returned.")
    if isinstance(frame.columns, pd.MultiIndex):
        tickers = frame.columns.get_level_values(-1)
        if ticker is None or ticker not in tickers:
            raise ValueError(f"The downloaded data does not contain {ticker}.")
        frame = frame.xs(ticker, axis=1, level=-1)

    if "Close" not in frame:
        raise ValueError("The input must include a Close column.")

    columns = [name for name in ("Open", "High", "Low", "Close", "Volume") if name in frame]
    result = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce")
    if result.index.tz is not None:
        result.index = result.index.tz_localize(None)
    result = result.loc[~result.index.isna()].sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    result = result.loc[np.isfinite(result["Close"]) & (result["Close"] > 0)]
    if result.empty:
        raise ValueError("No valid closing prices were available.")
    return result


def download_history(ticker, start, end):
    from yfinance import download

    if not TICKER_RE.fullmatch(ticker):
        raise ValueError("Enter a ticker such as MSFT, AAPL, or BRK-B.")
    if start >= end:
        raise ValueError("The end date must be after the start date.")
    return clean_history(
        download(ticker, start=start, end=end, interval="1d", auto_adjust=True,
                 progress=False, threads=False),
        ticker,
    )


def fit_scaler(close, end):
    """Fit only the supplied prefix; transform later observations separately."""
    values = np.asarray(close).reshape(-1, 1)
    return MinMaxScaler().fit(values[:end])


def windows_for_targets(values, targets, lookback=LOOKBACK):
    """Use exactly the observations preceding each target, not the target itself."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets, dtype=int)
    if len(targets) == 0 or np.any(targets < lookback) or np.any(targets >= len(values)):
        raise ValueError("Each target needs a full preceding window and an observed outcome.")
    starts = targets[:, None] - lookback + np.arange(lookback)
    return values[starts][..., None], values[targets]


def predict_prices(model, inputs, scaler):
    """Run bounded batches without creating a separate tf.data prediction pool."""
    batches = [np.asarray(model(inputs[i:i + 128], training=False)) for i in range(0, len(inputs), 128)]
    return scaler.inverse_transform(np.concatenate(batches)).ravel()