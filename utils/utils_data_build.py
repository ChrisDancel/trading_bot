import logging

log = logging.getLogger(__name__)

import requests
import pandas as pd
import pytz
import string
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import date
from utils import utils_shared as ush

today = datetime.today().astimezone(pytz.timezone("America/New_York"))
today_fmt = today.strftime("%Y-%m-%d")


def clean_symbol_data(df):
    log.debug("shape of df before removing nans: {}".format(df.shape))
    log.debug("removing nans...")
    df = df[df["closePrice"].notnull()].reset_index(drop=True)
    log.debug("shape of df after removing nans: {}".format(df.shape))
    log.debug("df preview: \n{}".format(df.head()))
    return df


class Ticker:
    def __init__(self):
        pass

    @staticmethod
    def _clean_symbol(symbol):
        return symbol.replace(".", "-").split("-")[0]

    @staticmethod
    def get_symbols():

        symbols = {}

        # Get a current list of all the stock symbols for the NYSE
        alpha = list(string.ascii_uppercase)

        for each in alpha[0:1]:
            log.info("alpha: {}".format(each))
            url = "http://eoddata.com/stocklist/NYSE/{}.htm".format(each)
            resp = requests.get(url)
            site = resp.content
            soup = BeautifulSoup(site, "html.parser")
            table = soup.find("table", {"class": "quotes"})
            for company_info in table.findAll("tr")[1:]:
                ticker_symbol = company_info.findAll("td")[0].text.rstrip()
                ticker_name = company_info.findAll("td")[1].text.rstrip()
                clean_ticker_symbol = Ticker._clean_symbol(ticker_symbol)
                symbols[clean_ticker_symbol] = ticker_name

        log.info(f"# symbols parsed: {len(symbols)}")
        return symbols


class DataStore:
    def __init__(self, conn, tablename):
        self.conn = conn
        self.tablename = tablename

    def get_ticker_unix_latest_date(self, ticker):

        try:
            db_date = pd.read_sql(
                "select max(datetime_unix) from stocks.{} where symbol = '{}'".format(
                    self.tablename, ticker
                ),
                self.conn,
            ).values[0][0]
        except:
            log.error("something bad happened..")
            db_date = None

        if db_date is None:
            start_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
            # Convert to unix for the API
            db_date = ush.unix_time_millis(start_date)

        return db_date

    def get_ticker_latest_date(self, ticker):

        db_date = self.get_ticker_unix_latest_date(ticker)

        return ush.unix_time_to_date_string(db_date)

    def get(self, ticker):
        return pd.read_sql(
            "select * from stocks.{} where symbol = '{}'".format(
                self.tablename, ticker
            ),
            self.conn,
        )

    def append(self, df):
        df.to_sql(con=self.conn, name=self.tablename, if_exists="append")

    def get_all(self):
        return pd.read_sql("select * from stocks.{}".format(self.tablename), self.conn)

    def get_tickers(self):
        return pd.read_sql(
            "select distinct(symbol) from stocks.{}".format(self.tablename), self.conn
        )


class DataSource:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_data(self, ticker, stock_history_params):
        """
        Makes an api call for a single stock and returns their historical trading positions.

        Input eg

        stock_history_params = {'periodType': 'month',
                           'period': 1,
                           'frequencyType': 'daily',
                           'frequency': 1}

        d = quotes_history_request('LEAF', **stock_history_params)

        """
        url = (
            r"https://api.tdameritrade.com/v1/marketdata/{stocks}/pricehistory".format(
                stocks=ticker
            )
        )
        params = {
            "apikey": self.api_key,
        }

        # merge params with kwargs
        params.update(stock_history_params)

        try:
            request = requests.get(url=url, params=params).json()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            log.error("requests error: {}".format(e))
            request = {"empty": True}

        try:
            if request["empty"]:
                log.error("error: cannot get information for symbol {}".format(ticker))
                df = pd.DataFrame(
                    dict(
                        open=[None],
                        low=[None],
                        closePrice=[None],
                        volume=[None],
                        datetime_unix=[1000],
                        date=[today_fmt],
                        symbol=[ticker],
                    )
                )
            else:
                df = pd.DataFrame.from_dict(request["candles"], orient="columns")
                df = df.rename(columns={"datetime": "datetime_unix"})
                df["date"] = df["datetime_unix"].apply(
                    lambda x: datetime.utcfromtimestamp(x / 1000).strftime("%Y-%m-%d")
                )
                df["symbol"] = ticker
                df = df.rename(columns={"close": "closePrice"})
            return df
            # time.sleep(0.1)
        except KeyError as e:
            log.error("error: {}".format(e))

    def get_deltas(self, ticker, start_date):
        stock_history_params = {
            "periodType": "year",
            "period": 3,
            "frequencyType": "daily",
            "frequency": 1,
            "startDate": start_date,
        }
        df = self.get_data(ticker=ticker, stock_history_params=stock_history_params)

        return df


class DataMaintenance:
    def __init__(self, data_store, data_source, ticker_list):
        self.data_store = data_store
        self.data_source = data_source
        self.ticker_list = ticker_list

    def update(self, ticker):

        latest_unix_date = self.data_store.get_ticker_unix_latest_date(ticker)
        latest_date = self.data_store.get_ticker_latest_date(ticker)

        log.info(f"latest date for ticker {ticker} is {latest_date}")

        today = date.today().strftime("%Y-%m-%d")
        date_increment = 60 * 60 * 24 * 1000
        tomorrow_unix_date = latest_unix_date + date_increment
        latest_date_plus_one_day = ush.unix_timestamp_to_date(tomorrow_unix_date)

        if today == latest_date_plus_one_day:
            log.info(
                "No need to update historical share prices - latest date is already last trading day {}!".format(
                    latest_date_plus_one_day
                )
            )
            return
        else:
            log.info(
                "Last date is {}; updating historical share prices...".format(
                    latest_date_plus_one_day
                )
            )

        deltas = self.data_source.get_deltas(ticker, start_date=tomorrow_unix_date)
        if deltas is not None:
            self.data_store.append(deltas)

    def update_all(self):
        log.info("Updating all tickers...")
        for t in self.ticker_list:
            self.update(t)
