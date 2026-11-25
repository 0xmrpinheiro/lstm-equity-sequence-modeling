"""Train a fresh LSTM and compare it with the previous close on a holdout."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_modeling import (
    LOOKBACK,
    clean_history,
    download_history,
    fit_scaler,
    predict_prices,
    sample_prices,
    windows_for_targets,
)


def prepare_holdout(close, lookback=LOOKBACK):
    """Split targets 70/15/15 by time and fit scaling on training data only."""
    count = len(close)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    if train_end <= lookback or validation_end <= train_end or validation_end >= count:
        raise ValueError(f"Not enough rows for a {lookback}-observation window and a 70/15/15 split.")

    scaler = fit_scaler(close, train_end)
    scaled = scaler.transform(close.to_numpy().reshape(-1, 1))
    train = windows_for_targets(scaled, np.arange(lookback, train_end), lookback)
    validation = windows_for_targets(scaled, np.arange(train_end, validation_end), lookback)
    test_indices = np.arange(validation_end, count)
    test = windows_for_targets(scaled, test_indices, lookback)
    return scaler, train_end, validation_end, train, validation, test_indices, test


def benchmark(close, epochs):
    import keras
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    keras.utils.set_random_seed(42)
    scaler, train_end, validation_end, train, validation, indices, test = prepare_holdout(close)
    model = keras.Sequential([
        keras.Input(shape=(LOOKBACK, 1)),
        keras.layers.LSTM(50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(50),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    def dataset(split):
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return tf.data.Dataset.from_tensor_slices(split).batch(32).with_options(options)

    model.fit(
        dataset(train), validation_data=dataset(validation), epochs=epochs, verbose=0, shuffle=False,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )],
    )

    values = close.to_numpy()
    actual = values[indices]
    predictions = {
        "LSTM": predict_prices(model, test[0], scaler),
        "Previous close": values[indices - 1],
    }
    print(f"Training ends: {close.index[train_end - 1]:%Y-%m-%d}")
    print(f"Validation ends: {close.index[validation_end - 1]:%Y-%m-%d}")
    print(f"Holdout: {close.index[indices[0]]:%Y-%m-%d} to {close.index[-1]:%Y-%m-%d} ({len(indices)} observations)")
    for name, forecast in predictions.items():
        mae = mean_absolute_error(actual, forecast)
        rmse = mean_squared_error(actual, forecast) ** 0.5
        print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", action="store_true", help="Use the synthetic offline data.")
    source.add_argument("--csv", type=Path, help="Read a CSV with Date and Close columns.")
    source.add_argument("--ticker", help="Download daily adjusted prices with yfinance.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2024-10-23", help="Exclusive download end date.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional path to save the newly trained model.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.output and (args.output.suffix != ".keras" or args.output.exists()):
        parser.error("--output must be a new .keras file")

    try:
        if args.sample:
            history = sample_prices()
            print("Source: synthetic sample (not market performance)")
        elif args.csv:
            history = clean_history(pd.read_csv(args.csv, index_col="Date"))
            print(f"Source: {args.csv}")
        else:
            ticker = args.ticker.strip().upper()
            start, end = pd.Timestamp(args.start).date(), pd.Timestamp(args.end).date()
            history = download_history(ticker, start, end)
            print(f"Source: Yahoo Finance adjusted daily closes for {ticker}")
        model = benchmark(history["Close"], args.epochs)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.output)


if __name__ == "__main__":
    main()