import logging

log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import os
import pandas_market_calendars as mcal
from prophet import Prophet
from datetime import timedelta, datetime


class Forecaster:
    def __init__(self, data_store, data_loc):
        self.data_store = data_store
        self.data_loc = data_loc

    @staticmethod
    def get_ticker_data_from_cache(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Return ticker data from a dataframe containing all ticker data - lazy function
        Args:
            df:
            symbol:

        Returns:
            df_sub
        """
        df_sub = (
            df[df["symbol"] == symbol]
                .sort_values(by="date")
                .drop("symbol", axis=1)
                .reset_index(drop=True)
        )
        return df_sub

    @staticmethod
    def add_sma(
            df: pd.DataFrame, col_value: str, col_date: str, sma_list=None
    ) -> pd.DataFrame:
        """
        Add simple moving average columns for a specified number of rolling average values

        Args:
            df: dataframe info on a SIINGLE ticker
            col_value: column name of stock price
            col_date: column name of stock date
            sma_list: list of sma values to compute

        Returns:
            df

        """
        if sma_list is None:
            sma_list = [10, 50, 200]
        df = df.sort_values([col_date], ascending=[True])

        # add sma
        for sma in sma_list:
            c_name = "ma_" + str(sma)
            df[c_name] = df.rolling(sma)[col_value].mean().reset_index(drop=True)

        return df

    @staticmethod
    def add_crossover(
            df: pd.DataFrame,
            short_col: str,
            long_col: str,
            col_name: str = "positions",
    ):
        """
        Add column <col_name> to show when short term signal crosses above long term signal
        Args:
            df: ticker history dataframe
            short_col: columne name of short signal
            long_col:  column name of long signal
            col_name: name of new column

        Returns:
            df

        """
        df["signal"] = np.where(df[short_col] > df[long_col], 1.0, 0.0)
        # Take the difference of the signals in order to generate actual trading orders
        df[col_name] = df["signal"].diff()
        df = df.drop(["signal"], axis=1)

        return df

    @staticmethod
    def add_days(str_date: str, days_to_add: int) -> str:
        """
        Add and return days to a date string
        Args:
            str_date:
            days_to_add:

        Returns:
            new_date

        """
        input_date = datetime.strptime(str_date, "%Y-%m-%d")
        new_date = input_date + timedelta(days=days_to_add)
        return new_date.strftime("%Y-%m-%d")

    def get_market_days_into_future(self, latest_date: str, days_to_add: int) -> list:
        """
        Get a list of future dates based on NYSE trading calendar
        Args:
            latest_date:
            days_to_add:

        Returns:
            future_dates

        """
        nyse = mcal.get_calendar("NYSE")

        end_date = self.add_days(latest_date, days_to_add)
        future_dates = nyse.schedule(start_date=latest_date, end_date=end_date)
        return future_dates.index[1:].strftime("%Y-%m-%d").to_list()

    def add_forecast(self, df: pd.DataFrame, days_into_future: int) -> pd.DataFrame:
        """
        Generate timeseries forecast using fbProphet for N days into the future

        Args:
            df: ticker dataframe
            days_into_future: number of days to forecast into the future

        Returns:
            forecast

        """
        mapper = {"date": "ds", "closePrice": "y"}
        inv_mapper = {v: k for k, v in mapper.items()}

        df_fb = df.rename(columns=mapper)[list(mapper.values())]

        m = Prophet(daily_seasonality=True)
        m.fit(df_fb)

        future = pd.DataFrame(df_fb["ds"]).append(
            pd.DataFrame(
                self.get_market_days_into_future(
                    df_fb["ds"][-1:].values[0], days_into_future
                ),
                columns=["ds"],
            ),
            ignore_index=True,
        )
        forecast = m.predict(future)
        forecast = forecast.rename(columns={"ds": "date", "yhat": "closePrice"})[
            list(mapper.keys())
        ]
        return forecast

    # def plot_ticker_data(self, df: pd.DataFrame):
    #     """
    #     Plot ticker data
    #     Args:
    #         df:
    #
    #     Returns:
    #
    #     """
    #     plt.figure(figsize=(15, 15))
    #     plt.plot(df.date, df.closePrice)
    #     plt.tight_layout()

    def get_future_signals(
            self,
            df_all_tickers: pd.DataFrame,
            ticker: str,
            days_to_forecast: int,
            long_signal: str = "50_200",
            short_signal: str = "10_50",
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Get future buy/sell signals for a given ticker based on SMA crossover positions.

        Method
        1). From a dataframe containing all ticker data history, get data for a selected ticker
        2). Add forecasts up to <days_to_forecast>
        3). add SMAs for long and short signals
        4). add crossover positions
        5). split analysis into dataframe for future buy and future sell dates

        Args:
            df_all_tickers: dataframe containing all ticker history
            ticker: specific ticker
            days_to_forecast: number of days to forecast into the future
            long_signal: '50_200' e.g. when SMA_50 positively crosses SMA_200
            short_signal: '10_50' e.g. when SMA_10 negatively crosses SMA_50

        Returns:
            df_buy_hist: buy signals
            df_sell_hist: sell signals

        """
        df = self.get_ticker_data_from_cache(df_all_tickers, ticker)

        latest_date = df["date"].max()
        df = self.add_forecast(df, days_into_future=days_to_forecast)

        select_long_signal = long_signal
        select_short_signal = short_signal

        df = self.add_sma(df, "closePrice", "date")

        long_column_name = (
                "positions_"
                + select_long_signal.split("_")[0]
                + "_"
                + select_long_signal.split("_")[1]
        )
        df = self.add_crossover(
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
        df = self.add_crossover(
            df,
            short_col="ma_" + select_short_signal.split("_")[0],
            long_col="ma_" + select_short_signal.split("_")[1],
            col_name=short_column_name,
        )
        df["ticker"] = ticker

        # get dataframes of past buy and sell signals
        df_buy_hist = df.loc[(df[long_column_name] == 1.0) & (df["date"] > latest_date)]
        df_sell_hist = df.loc[
            (df[short_column_name] == -1.0) & (df["date"] > latest_date)
            ]

        cols_to_not_show = [long_column_name, short_column_name]

        df_buy_hist = df_buy_hist[df_buy_hist.columns.difference(cols_to_not_show)]
        df_sell_hist = df_sell_hist[df_sell_hist.columns.difference(cols_to_not_show)]

        return df_buy_hist, df_sell_hist

    def collect_all_future_signals(
            self, df: pd.DataFrame, symbols: list, days_to_forecast: int
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Collect all future buy/sell signals for all tickers specified in <symbols>

        Args:
            df: dataframe containing all ticker history
            symbols: list of symbols to collect future signals for
            days_to_forecast: number of days to forecast into the futurex

        Returns:
            df_buy_hist_all
            df_sell_hist_all

        """

        df_buy_hist_all = pd.DataFrame()
        df_sell_hist_all = pd.DataFrame()

        assert os.path.exists(
            self.data_loc
        ), f"error: data path {self.data_loc} doesnt exist - pls double check"

        buy_save_path = os.path.join(self.data_loc, "df_buy_hist")
        sell_save_path = os.path.join(self.data_loc, "df_sell_hist")

        for n, s in enumerate(symbols):

            log.info(f"{n}: symbol {s}")

            try:
                df_buy, df_sell = self.get_future_signals(
                    df_all_tickers=df, ticker=s, days_to_forecast=days_to_forecast
                )
            except ValueError as e:
                log.error(f"cannot forecast for ticker {s}: {e}")
                df_buy = pd.DataFrame()
                df_sell = pd.DataFrame()

            df_buy_hist_all = df_buy_hist_all.append(df_buy)
            df_sell_hist = df_sell_hist_all.append(df_sell)

            # save every n times just in case programme errors out after a long run with no data to show for its efforts
            if n % 50 == 0:
                df_buy_hist_all.to_pickle(buy_save_path)
                df_sell_hist.to_pickle(sell_save_path)

        df_buy_hist_all.to_pickle(buy_save_path)
        df_sell_hist_all.to_pickle(sell_save_path)

        return df_buy_hist_all, df_sell_hist_all
