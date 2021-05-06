import logging

log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import json
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from utils import utils_data_build as udb
from utils import utils_data_forecaster as udf
from utils import utils_email as ue
from utils import utils_shared as ush

markdown_text = """
This application is an extension of the Recommender Trading Bot in that it helps the user to visualise previous prices,
future forecast prices, and any buy/sell signals. 
"""


@st.cache
def get_data():
    return data_store.get_all()


@st.cache
def get_symbol_data():
    return data_store.get_tickers()


@st.cache
def get_symbol_names():
    tickers = udb.Ticker()
    symbols = tickers.get_symbols()
    return [str(k + ", " + v) for k, v in symbols.items()]


def pct_change(from_value, to_value):
    return (to_value - from_value) / from_value


def main(ds, fc, mailer):
    st.write("# Stock Analysis App")
    st.markdown(markdown_text)

    # list all symbol names in format 'ABC, alpha beta gamma company'
    symbol_names = get_symbol_names()

    # ------  SIDEBAR GENERATION ----------
    st.sidebar.markdown('### Parameter Selection')

    select_symbol_description = st.sidebar.selectbox(f"Stock Name", tuple(symbol_names))
    select_long_signal = st.sidebar.selectbox(f"Long Signal", ("50_200", "10_50"))
    select_short_signal = st.sidebar.selectbox(f"Short Signal", ("50_200", "10_50"))

    select_history = st.sidebar.radio(
        "History", ("all", "1y", "6m", "3m", "1m", "7d", "1d")
    )
    # ------  END SIDEBAR GENERATION ----------

    select_symbol = select_symbol_description.split(",")[0]  # parse ticker
    select_name = select_symbol_description.split(",")[1]  # parse company name

    # st.write(f"long signal selected: {select_long_signal}")
    # st.write(f"Short signal selected: {select_short_signal}")

    df = ds.get(select_symbol)
    df_sorted = df.sort_values(by="date", ascending=False)

    daily_price_diff = df_sorted.iloc[0]["closePrice"] - df_sorted.iloc[0]["open"]
    daily_price_pct_diff = daily_price_diff / df_sorted.iloc[0]["open"]

    f"## {select_name} ({select_symbol})"
    f"### open: {df_sorted.iloc[0]['open']}p | close: {df_sorted.iloc[0]['closePrice']}p | {daily_price_diff:.2f}p ({daily_price_pct_diff * 100:.2f}%)"

    df_info = pd.DataFrame(
        {
            "symbol": [select_symbol],
            "price": [df_sorted.iloc[0]["closePrice"]],
            "previous close": [df_sorted.iloc[1]["closePrice"]],
            "change": [
                pct_change(
                    df_sorted.iloc[1]["closePrice"], df_sorted.iloc[0]["closePrice"]
                )
            ],
            "Trade high": [df_sorted.iloc[0]["high"]],
            "Trade low": [df_sorted.iloc[0]["low"]],
            "Volume": [df_sorted.iloc[0]["volume"]],
        }
    )

    st.dataframe(df_info)
    latest_date = df["date"].max()
    df = forecaster.add_forecast(df, days_into_future=31)

    len_future = np.sum(df["date"] > latest_date)

    df = fc.add_sma(df, "closePrice", "date")

    long_column_name = (
            "positions_"
            + select_long_signal.split("_")[0]
            + "_"
            + select_long_signal.split("_")[1]
    )
    df = fc.add_crossover(
        df,
        short_col="ma_" + select_long_signal.split("_")[0],
        long_col="ma_" + select_long_signal.split("_")[1],
        col_name=long_column_name,
    )

    short_column_name = (
            "positions_"
            + select_short_signal.split("_")[0]
            + "_"
            + select_short_signal.split("_")[1]
    )
    df = fc.add_crossover(
        df,
        short_col="ma_" + select_short_signal.split("_")[0],
        long_col="ma_" + select_short_signal.split("_")[1],
        col_name=short_column_name,
    )

    # adding section here to get dates
    # get dataframes of past buy and sell signals
    # df_buy_hist = df.loc[df[long_column_name] == 1.0]
    # df_sell_hist = df.loc[df[short_column_name] == -1.0]

    df = df.rename(columns={"date": "index"}).set_index("index")
    fig, ax1 = plt.subplots(figsize=(20, 10))

    cols_to_not_show = [long_column_name, short_column_name]

    select_history_mapper = {
        "all": 1000,
        "1y": 365,
        "6m": 183,
        "3m": 90,
        "1m": 31,
        "7d": 7,
        "1d": 1,
    }

    days_to_subtract = select_history_mapper[select_history]

    if short_column_name in df.columns:

        current = np.datetime64("today")
        min_date = current - np.timedelta64(days_to_subtract, "D")
        dfx = df[df.index > min_date]

        ax1.plot(dfx[dfx.columns.difference(cols_to_not_show)])
        ax1.axvline(x=current, color="k", linestyle="--")

        # Plot the "buy" trades
        ax1.plot(
            dfx.loc[dfx[long_column_name] == 1.0].index,
            dfx["ma_" + select_long_signal.split("_")[0]][dfx[long_column_name] == 1.0],
            "^",
            markersize=10,
            color="m",
        )

        # Plot the "sell" trades
        ax1.plot(
            dfx.loc[dfx[short_column_name] == -1.0].index,
            dfx["ma_" + select_short_signal.split("_")[0]][
                dfx[short_column_name] == -1.0
                ],
            "v",
            markersize=10,
            color="k",
        )
        ax1.legend(dfx.columns.difference(cols_to_not_show))
        plt.xticks(rotation=45, fontsize=32)
        plt.yticks(fontsize=32)
        plt.grid()
        plt.tight_layout()

    else:
        plt.plot(df)

    st.pyplot(fig)

    "### Buy signals"
    st.dataframe(mailer.get_email_data("data/df_buy_hist"))

    "### Sell signals"
    st.dataframe(mailer.get_email_data("data/df_sell_hist"))


if __name__ == "__main__":
    with open("configs/config.json") as f:
        config = json.load(f)

    log.info("Generating Data Store object...")
    conn = ush.create_conn(**config["MYSQL"])
    tablename_historical_prices = config["MYSQL"]["tablename_historical_data"]
    data_store = udb.DataStore(conn, tablename_historical_prices)

    forecaster = udf.Forecaster(data_store=data_store, data_loc="data")

    n_days_max = config["EMAIL"]["n_days_max"]
    mail = ue.Mail(
        from_email="",
        to_email="",
        from_pw="",
        n_days_max=n_days_max,
        data_loc="data",
        df_buy_name="df_buy_hist",
        df_sell_name="df_sell_hist",
    )

    main(ds=data_store, fc=forecaster, mailer=mail)
