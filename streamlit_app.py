"""Offline-first demonstration of LSTM sequence modeling, not a benchmark."""

from datetime import date, timedelta

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sequence_modeling import (
    LOOKBACK,
    TICKER_RE,
    download_history,
    fit_scaler,
    load_model,
    predict_prices,
    sample_prices,
    windows_for_targets,
)


@st.cache_resource(show_spinner=False)
def demo_model():
    return load_model()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_prices(ticker, start, end):
    return download_history(ticker, start, end)


def show_history(frame):
    close = frame["Close"]
    fig = go.Figure(go.Scatter(x=frame.index, y=close, name="Close"))
    for days in (100, 200):
        if len(close) >= days:
            fig.add_scatter(x=frame.index, y=close.rolling(days).mean(), name=f"{days}-observation mean")
    fig.update_layout(title="Closing prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, width="stretch", key="prices")

    ohlc = ["Open", "High", "Low", "Close"]
    if all(column in frame for column in ohlc):
        candles = frame.dropna(subset=ohlc)
        if not candles.empty:
            fig = go.Figure(go.Candlestick(
                x=candles.index, open=candles["Open"], high=candles["High"],
                low=candles["Low"], close=candles["Close"],
            ))
            fig.update_layout(title="Adjusted candlesticks", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, width="stretch", key="candles")
    if "Volume" in frame:
        fig = go.Figure(go.Bar(x=frame.index, y=frame["Volume"]))
        fig.update_layout(title="Reported volume", xaxis_title="Date")
        st.plotly_chart(fig, width="stretch", key="volume")


def show_replay(frame):
    close = frame["Close"]
    if len(close) <= LOOKBACK:
        st.info("The replay needs at least 101 closing observations.")
        return

    st.subheader("Historical model replay")
    st.caption(
        "Illustration only. The bundled model is reused as-is, and this chart is not a "
        "holdout evaluation. For an apples-to-apples comparison with the previous close, "
        "run the separate benchmark from the repository."
    )
    start = max(LOOKBACK, int(len(close) * 0.70), len(close) - 200)
    targets = np.arange(start, len(close))
    scaler = fit_scaler(close, start)
    scaled = scaler.transform(close.to_numpy().reshape(-1, 1))
    inputs, _ = windows_for_targets(scaled, targets)
    with st.spinner("Running the bundled LSTM..."):
        model = demo_model()
        predictions = predict_prices(model, inputs, scaler)
    fig = go.Figure()
    fig.add_scatter(x=close.index[targets], y=close.iloc[targets], name="Observed close")
    fig.add_scatter(x=close.index[targets], y=predictions, name="LSTM replay")
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, width="stretch", key="replay")


def main():
    st.set_page_config(page_title="LSTM Equity Sequence Modeling", layout="wide")
    st.title("LSTM Equity Sequence Modeling")
    st.caption("Explore price sequences and a bundled LSTM. Educational demo, not investment advice.")
    source = st.sidebar.radio("Data source", ["Synthetic sample (offline)", "Yahoo Finance"])
    if source == "Synthetic sample (offline)":
        st.info("Showing a generated price series. These are not prices for a real ticker.")
        frame = sample_prices()
    else:
        with st.sidebar.form("history"):
            ticker = st.text_input("Ticker", "MSFT").strip().upper()
            end = date.today()
            start = st.date_input("Start date", end - timedelta(days=365 * 3))
            end = st.date_input("End date (exclusive)", end)
            submitted = st.form_submit_button("Load history")
        if submitted:
            st.session_state.pop("downloaded_history", None)
            if not TICKER_RE.fullmatch(ticker) or start >= end:
                st.error("Enter a valid ticker and an end date after the start date.")
                return
            try:
                with st.spinner("Fetching adjusted daily prices..."):
                    st.session_state["downloaded_history"] = (ticker, fetch_prices(ticker, start, end))
            except Exception:
                st.error("Couldn't load prices. Check the ticker, date range, and network or try the offline sample.")
                return
        if "downloaded_history" not in st.session_state:
            st.info("Choose a ticker and select Load history, or try the offline sample.")
            return
        loaded_ticker, frame = st.session_state["downloaded_history"]
        st.subheader(f"{loaded_ticker} · adjusted daily data")

    show_history(frame)
    try:
        show_replay(frame)
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Could not run the bundled Keras model: {error}")


if __name__ == "__main__":
    main()