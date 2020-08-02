import os
import ast
import requests
import pandas as pd
import numpy as np
import pytz
from bs4 import BeautifulSoup
import string
import time
from datetime import datetime
from pandas.io import sql
from sqlalchemy import create_engine
# from google.cloud import bigquery
# from google.cloud import storage
# from google.cloud.exceptions import NotFound

from matplotlib import pyplot as plt
from scipy import stats

import alpaca_trade_api as tradeapi

# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# from oauth2client.client import GoogleCredentials

# path_to_creds = os.path.join(os.getcwd(), 'configs', 'trading-bot-3dabe112fe73.json')
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path_to_creds

today = datetime.today().astimezone(pytz.timezone("America/New_York"))
today_fmt = today.strftime('%Y-%m-%d')


# def get_config_from_gcp(bucket_name, file_name):
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(file_name)
#     api_key = blob.download_as_string()
#     config = ast.literal_eval(api_key.decode())

#     return config


def get_symbols(exchange):
    # Get a current list of all the stock symbols for the NYSE
    # Create a list of every letter in the alphabet
    # Each page has a letter for all those symbols
    # i.e. http://eoddata.com/stocklist/NYSE/A.htm'
    alpha = list(string.ascii_uppercase)

    symbols = []

    # Loop through the letters in the alphabet to get the stocks on each page
    # from the table and store them in a list

    for each in alpha:
        url = 'http://eoddata.com/stocklist/{exchange}/{each}.htm'.format(exchange=exchange, each=each)
        resp = requests.get(url)
        site = resp.content
        soup = BeautifulSoup(site, 'html.parser')
        table = soup.find('table', {'class': 'quotes'})
        for row in table.findAll('tr')[1:]:
            symbols.append(row.findAll('td')[0].text.rstrip())

            # Remove the extra letters on the end
    symbols_clean = []

    for each in symbols:
        each = each.replace('.', '-')
        symbols_clean.append((each.split('-')[0]))

    return symbols_clean


# The TD Ameritrade api has a limit to the number of symbols you can get data for
# in a single call so we chunk the list into 200 symbols at a time 
def chunks(n_list, n):
    """
    Takes in a list and how long you want
    each chunk to be
    """
    n = max(1, n)
    n_set = (n_list[i:i + n] for i in range(0, len(n_list), n))
    return list(n_set)


# Function for the api request to get the data from td ameritrade
def quotes_request(stocks, api_key):
    """
    Makes an api call for a list of stock symbols
    and returns a dataframe
    """
    url = r"https://api.tdameritrade.com/v1/marketdata/quotes"

    params = {
        'apikey': api_key,
        'symbol': stocks
    }

    request = requests.get(
        url=url,
        params=params
    ).json()

    time.sleep(0.1)

    return pd.DataFrame.from_dict(
        request,
        orient='index'
    ).reset_index(drop=True)


def quotes_history_request(stocks, api_key, **stock_history_params):
    """
    Makes an api call for a single stock and returns their historical trading positions. 
    
    Input eg
    
    stock_history_params = {'periodType': 'month', 
                       'period': 1, 
                       'frequencyType': 'daily', 
                       'frequency': 1}

    d = quotes_history_request('LEAF', **stock_history_params)

    """
    url = r"https://api.tdameritrade.com/v1/marketdata/{stocks}/pricehistory".format(stocks=stocks)
    params = {
        'apikey': api_key,
    }

    # merge params with kwargs
    params.update(stock_history_params)

    request = requests.get(
        url=url,
        params=params
    ).json()

    try:
        if request['empty']:
            print('error: cannot get information for symbol {}'.format(stocks))
            df = pd.DataFrame(dict(open=[None],
                                   low=[None],
                                   closePrice=[None],
                                   volume=[None],
                                   datetime_unix=[1000],
                                   date=[today_fmt],
                                   symbol=[stocks]))
        else:
            df = pd.DataFrame.from_dict(request['candles'], orient='columns')
            df = df.rename(columns={'datetime': 'datetime_unix'})
            df['date'] = df['datetime_unix'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d'))
            df['symbol'] = stocks
            df = df.rename(columns={'close': 'closePrice'})
        return df
        time.sleep(0.1)
    except KeyError as e:
        print('error: {}'.format(e))
        


def clean_quotes_data(symbols_chunked, api_key):
    # Check if the market was open today. Cloud functions use UTC and I'm in
    # eastern so I convert the timezone
    today = datetime.today().astimezone(pytz.timezone("America/New_York"))
    today_fmt = today.strftime('%Y-%m-%d')

    df = pd.concat([quotes_request(each, api_key) for each in symbols_chunked])

    # Add the date and fmt the dates for BQ
    df['date'] = pd.to_datetime(today_fmt)
    df['date'] = df['date'].dt.date
    df['divDate'] = pd.to_datetime(df['divDate'])
    df['divDate'] = df['divDate'].dt.date
    df['divDate'] = df['divDate'].fillna(np.nan)

    # Remove anything without a price
    df = df.loc[df['bidPrice'] > 0]

    # Rename columns and format for bq (can't start with a number)
    df = df.rename(columns={
        '52WkHigh': '_52WkHigh',
        '52WkLow': '_52WkLow'
    })

    return df


def clean_historical_quotes_data(symbols_chunked, api_key, stock_history_params):
    df = pd.DataFrame()

    for i, s in enumerate(symbols_chunked):

        if i % 10 == 0:
            print('{} of {} - symbol {}'.format(i, len(symbols_chunked), s))

        df_sub = quotes_history_request(stocks=s,
                                        api_key=api_key,
                                        **stock_history_params)
        df = df.append(df_sub)

    return df


# def load_to_bigquery(df, dataset_id, table_id):
#     # Add to bigquery
#     client = bigquery.Client()

#     dataset_ref = client.dataset(dataset_id)
#     table_ref = dataset_ref.table(table_id)

#     job_config = bigquery.LoadJobConfig()
#     job_config.source_format = bigquery.SourceFormat.CSV
#     job_config.autodetect = True
#     job_config.ignore_unknown_values = True
#     job = client.load_table_from_dataframe(
#         df,
#         table_ref,
#         #             location='US',
#         job_config=job_config
#     )

#     job.result()

#     print('*** Load to bigquery finished! ***')


# def get_latest_unix_date_bq(dataset_id, table_id):
#     # Add to bigquery
#     client = bigquery.Client()

#     sql = """
#     select max(datetime_unix) as max_datetime_unix
#     from `{}.{}`     
#     """.format(dataset_id, table_id)

#     return client.query(sql).to_dataframe()['max_datetime_unix'][0]


    
    
def unix_timestamp_to_date(uts):

    # check if uts is in milliseconds
    if len(str(uts)) == 13:
        return datetime.utcfromtimestamp(uts / 1000).strftime('%Y-%m-%d')
    elif len(str(uts)) == 10:
        return datetime.utcfromtimestamp(uts).strftime('%Y-%m-%d')
    else:
        raise ValueError('unix timestamp input looks wrong: {}'.format(uts))


# def check_gbq_table_exists(client, table_ref):

#     try:
#         client.get_table(table_ref)
#         return True
#     except NotFound:
#         return False


# def get_historical_stockprices(config):
#     client = bigquery.Client()

#     # Load the historical stock data from BQ
#     sql_hist = """
#         SELECT
#           symbol,
#           closePrice,
#           date
#         FROM 
#           `{}.{}.{}`
#         """.format(config['BIGQUERY']['project_id'],
#                    config['BIGQUERY']['dataset_id'],
#                    config['BIGQUERY']['historical_table_id'])

#     df = client.query(sql_hist).to_dataframe()

#     # Convert the date column to datetime
#     # df['date'] = pd.to_datetime(df['date'])

#     # Sort by date (ascending) for the momentum calculation
#     df = df.sort_values(by='date').reset_index(drop=True)

#     # Rename the column
#     df = df.rename(columns={'closePrice': 'close'})

#     return df


def calc_portfolio_value(config):

    # Initialize the alpaca api
    api = tradeapi.REST(
        config['ALPACA']['alpaca_api_key'],
        config['ALPACA']['alpaca_secret_key'],
        config['ALPACA']['base_url'],
        'v2'
    )

    # Get the current positions from alpaca and create a df
    positions = api.list_positions()

    # print('positions: {}'.format(positions))

    symbol, qty, market_value = [], [], []

    for each in positions:
        symbol.append(each.symbol)
        qty.append(int(each.qty))
        market_value.append(float(each.market_value))

    df_pf = pd.DataFrame(
        {
            'symbol': symbol,
            'qty': qty,
            'market_value': market_value
        }
    )

    portfolio_value = round(df_pf['market_value'].sum(), 2)
    return portfolio_value, df_pf


# Momentum score function
def momentum_score(ts):
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    regress = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(regress[0]), 252) -1) * 100
    return annualized_slope * (regress[2] ** 2)


# Function to get the momentum stocks we want
def get_momentum_stocks(df, date, portfolio_size, cash):
    # Filter the df to get the top 10 momentum stocks for the latest day
    #     df_top_m = df.loc[df['date'] == pd.to_datetime(date)]
    #     df_top_m = df_top_m.sort_values(by='momentum', ascending=False).head(portfolio_size)
    df_top_m = df.loc[df['date'] == date].sort_values(by='momentum', ascending=False).head(portfolio_size)

    # print('df_top_m: {}'.format(df_top_m))

    # Set the universe to the top momentum stocks for the period
    universe = df_top_m['symbol'].tolist()

    # print('universe: {}'.format(universe))

    # Create a df with just the stocks from the universe
    df_u = df.loc[df['symbol'].isin(universe)]

    # print('df_u 1: {}'.format(df_u))

    # Create the portfolio
    # Pivot to format for the optimization library
    df_u = df_u.pivot_table(
        index='date',
        columns='symbol',
        values='close',
        aggfunc='sum'
    )

    # print('df_u 2: {}'.format(df_u))

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df_u)
    # print('mu: {}'.format(mu))

    S = risk_models.sample_cov(df_u)
    # print('S: {}'.format(S))

    # Optimise the portfolio for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S, gamma=1)  # Use regularization (gamma=1)
    # print('ef: {}'.format(ef))

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    # print('cleaned_weights: \n{}'.format(cleaned_weights))

    # Allocate
    latest_prices = get_latest_prices(df_u)
    # print('latest_prices: \n{}'.format(latest_prices))

    da = DiscreteAllocation(
        cleaned_weights,
        latest_prices,
        total_portfolio_value=cash
    )
    # print('da: \n{}'.format(da))

    allocation = da.lp_portfolio()[0]
    # print('allocation: \n{}'.format(allocation))

    # Put the stocks and the number of shares from the portfolio into a df
    symbol_list = []
    num_shares_list = []

    for symbol, num_shares in allocation.items():
        symbol_list.append(symbol)
        num_shares_list.append(num_shares)

    # Now that we have the stocks we want to buy we filter the df for those ones
    df_buy = df.loc[df['symbol'].isin(symbol_list)]
    # print('df_buy 1: \n{}'.format(df_buy))

    # Filter for the period to get the closing price
    df_buy = df_buy.loc[df_buy['date'] == date].sort_values(by='symbol')
    # print('df_buy 2: \n{}'.format(df_buy))

    # Add in the qty that was allocated to each stock
    df_buy['qty'] = num_shares_list
    # print('df_buy 3: \n{}'.format(df_buy))

    # Calculate the amount we own for each stock
    df_buy['amount_held'] = df_buy['close'] * df_buy['qty']
    df_buy = df_buy.loc[df_buy['qty'] != 0]
    # print('df_buy 4: \n{}'.format(df_buy))

    return df_buy


def sell_stocks(df, df_pf, sell_list, date):
    # Get the current prices and the number of shares to sell
    df_sell_price = df.loc[df['date'] == date]

    # Filter
    df_sell_price = df_sell_price.loc[df_sell_price['symbol'].isin(sell_list)]

    # Check to see if there are any stocks in the current ones to buy
    # that are not in the current portfolio. It's possible there may not be any
    if df_sell_price.shape[0] > 0:
        df_sell_price = df_sell_price[[
            'symbol',
            'close'
        ]]

        # Merge with the current pf to get the number of shares we bought initially
        # so we know how many to sell
        df_buy_shares = df_pf[[
            'symbol',
            'qty'
        ]]

        df_sell = pd.merge(
            df_sell_price,
            df_buy_shares,
            on='symbol',
            how='left'
        )

    else:
        df_sell = None

    return df_sell


# Get a list of all stocks to sell i.e. any not in the current df_buy and any diff in qty
def stock_diffs(df_sell, df_pf, df_buy):
    # Select only the columns we need
    df_stocks_held_prev = df_pf[['symbol', 'qty']]
    df_stocks_held_curr = df_buy[['symbol', 'qty', 'date', 'close']]

    # Inner merge to get the stocks that are the same week to week
    df_stock_diff = pd.merge(
        df_stocks_held_curr,
        df_stocks_held_prev,
        on='symbol',
        how='inner'
    )

    # Calculate any difference in positions based on the new pf
    df_stock_diff['share_amt_change'] = df_stock_diff['qty_x'] - df_stock_diff['qty_y']

    # Create df with the share difference and current closing price
    df_stock_diff = df_stock_diff[[
        'symbol',
        'share_amt_change',
        'close'
    ]]

#     print('df_stock_diff: {}'.format(df_stock_diff))
    # If there's less shares compared to last week for the stocks that
    # are still in our portfolio, sell those shares
    df_stock_diff_sale = df_stock_diff.loc[df_stock_diff['share_amt_change'] < 0]

    # If there are stocks whose qty decreased,
    # add the df with the stocks that dropped out of the pf
#     print('df_stock_diff_sale: {}'.format(df_stock_diff_sale))

    if df_stock_diff_sale.shape[0] > 0:
        if df_sell is not None:
            df_sell_final = pd.concat([df_sell, df_stock_diff_sale])
            # Fill in NaNs in the share amount change column with
            # the qty of the stocks no longer in the pf
            df_sell_final['share_amt_change'] = df_sell_final['share_amt_change'].fillna(df_sell_final['qty'])
            # Turn the negative numbers into positive for the order
            df_sell_final['share_amt_change'] = np.abs(df_sell_final['share_amt_change'])
            # remove extra 'qty' column
            df_sell_final = df_sell_final.drop('qty', axis=1)
            df_sell_final = df_sell_final.rename(columns={'share_amt_change': 'qty'})
        else:
            df_sell_final = df_stock_diff_sale
            # Turn the negative numbers into positive for the order
            df_sell_final['share_amt_change'] = np.abs(df_sell_final['share_amt_change'])
            df_sell_final = df_sell_final.rename(columns={'share_amt_change': 'qty'})
    elif df_sell is not None:
            df_sell_final = df_sell
    else:
        df_sell_final = None

    return df_sell_final


def buy_new_stock(df_pf, df_buy):
    """
    Buy the stocks that increased in shares compared to last week or any new stocks


    """
    # Left merge to get any new stocks or see if they changed qty
    df_buy_new = pd.merge(
        df_buy,
        df_pf,
        on='symbol',
        how='left'
        )

    # Get the qty we need to increase our positions by
    df_buy_new = df_buy_new.fillna(0)
    df_buy_new['qty_new'] = df_buy_new['qty_x'] - df_buy_new['qty_y']

    # Filter for only shares that increased
    df_buy_new = df_buy_new.loc[df_buy_new['qty_new'] > 0]
    if df_buy_new.shape[0] > 0:
        df_buy_new = df_buy_new[[
            'symbol',
            'qty_new',
            'close'
        ]]
        df_buy_new = df_buy_new.rename(columns={'qty_new': 'qty'})
    else:
        df_buy_new = None

    return df_buy_new


def order_stock(df, api, side):
    """
    alpaca api wrapper to either buy or sell stock described in dataframe df

    df (dataframe): list of stocks
    api (object): api to alpaca
    side (str): 'buy' or 'sell'
    """

    assert side in ['buy', 'sell'], 'incorrect input for side {}'.format(side)

    # Send the sell order to the api
    if df is not None:
        symbol_list = df['symbol'].tolist()
        qty_list = df['qty'].tolist()
        try:
            for symbol, qty in list(zip(symbol_list, qty_list)):
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
        except Exception:
            pass


def portfolio_add_stock(df_pf, df_buy_stock):
    return df_pf.append(df_buy_stock).groupby(['symbol'])['qty'].agg('sum').reset_index()


def portfolio_remove_stock(df_pf, df_sell_stock):
    df_sell_stock['qty'] *= -1
    return df_pf.append(df_sell_stock).groupby(['symbol'])['qty'].agg('sum').reset_index()


def compute_pf_value_at_date(df_pf, df_stock_hist, date):
    date_filt = df_stock_hist['date'] == date
    df_stock_hist_f1 = df_stock_hist[date_filt]

    symbol_filt = [True if i in list(df_pf['symbol']) else False for i in df_stock_hist_f1['symbol']]
    df_stock_hist_f2 = df_stock_hist_f1[symbol_filt]
    #     df_pf_todays_price = df_stock_hist[date_filt & symbol_filt][['symbol', 'close']]

    df_pf_todays_price = df_stock_hist_f2[['symbol', 'close']]

    df_pf_with_price = pd.merge(df_pf, df_pf_todays_price, on='symbol', how='inner')
    df_pf_with_price['market_value'] = df_pf_with_price['qty'] * df_pf_with_price['close']

    portfolio_value_at_date = df_pf_with_price['market_value'].sum()

    return portfolio_value_at_date, df_pf_with_price


def create_conn(**creds_json):
    engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user=creds_json['username'],
                               pw=creds_json['password'],
                               db=creds_json['database']))
    
    return engine
        
def load_to_mysql(df, tablename, connection, if_exists='replace'):
    df.to_sql(con=connection, name=tablename, if_exists=if_exists)
    
    
def read_sql_to_df(sql: str, connection):
    return pd.read_sql(sql, connection);


def check_table_exists(conn, tablename):
    """
    check if a table exists in database
    """
    return True if tablename in conn.table_names() else False

def get_latest_unix_date(conn, tablename):
    """
    get latest timestamp in unix format from a particular table
    """
    return pd.read_sql("select max(datetime_unix) from stocks.{}".format(tablename), conn).values[0][0]
    
    
# def add_sma(df: pd.DataFrame, 
#             col_groupby: str, 
#             col_value: str,
#             col_date: str, 
#             sma_list: list=[20, 50, 200]) -> pd.DataFrame:
    
#     # sort by date
# #     df = df.sort_values(by=col_date, ascending=True).reset_index(drop=True)
#     df = df.sort_values([col_groupby, col_date], ascending=[True, True])
    
#     # add sma
#     # smas = [5, 10, 50, 100, 200]
#     for sma in sma_list:
#         c_name = 'ma_'+str(sma)
#         df[c_name] = df.groupby(col_groupby).rolling(sma)[col_value].mean().reset_index(drop=True)
        
#     return df


# get crossover points
def crossover(df, col1, col2, col_name):
    df['df1'] = df[col1]-df[col2]
    df['df1_yest'] = df['df1'].shift(1)
    ind_pos_cross = (df['df1']>0) & (df['df1_yest'] < 0)
    ind_neg_cross = (df['df1']<0) & (df['df1_yest'] > 0)
    
    df[col_name] = 0
    df.loc[ind_pos_cross, col_name] = 1
    df.loc[ind_neg_cross, col_name] = -1
    df = df.drop(['df1', 'df1_yest'], axis=1)
    return df


def add_crossover_pairs(df):
    
    # TODO - this list is hardcoded - is there a better way to code this?
    crossover_pairs = [[5, 10], 
                      [10, 20], 
                      [10, 50], 
                      [10, 100], 
                      [20, 50], 
                       [20, 100], 
                       [50, 100]]
    
    for p in crossover_pairs:
        df = utb.crossover(df, 'ma_{}'.format(p[0]), 'ma_{}'.format(p[1]), 'crossover_{}_{}'.format(p[0], p[1]))

    return df


class Visualiser():
    
    def __init__(self, conn):
        self.conn = conn
        
    def get_all_data(self):
        sql = "select symbol, date,closePrice from stocks.historical_prices"
        return read_sql_to_df(sql=sql, connection=self.conn)
    
    def get_ticker_data(self, symbol):
        sql = f"select symbol, date, closePrice from stocks.historical_prices where symbol = '{symbol}'"
        return read_sql_to_df(sql=sql, connection=self.conn)
    
    def get_ticker_data_from_cache(self, df, symbol):
        df_sub = df[df['symbol'] == symbol]\
            .sort_values(by='date')\
            .drop('symbol', axis=1)\
            .reset_index(drop=True)
        return df_sub

    def get_all_symbols(self):
        sql = f"select distinct symbol from stocks.historical_prices"
        return read_sql_to_df(sql=sql, connection=self.conn)        
        

    def add_sma(self, df: pd.DataFrame, 
                col_value: str,
                col_date: str, 
                sma_list: list=[50, 200]) -> pd.DataFrame:

        df = df.sort_values([col_date], ascending=[True])

        # add sma
        for sma in sma_list:
            c_name = 'ma_'+str(sma)
            df[c_name] = df.rolling(sma)[col_value].mean().reset_index(drop=True)

        return df

    def add_crossover(self, df, short_col, long_col):
        df['signal'] = np.where(df[short_col] > df[long_col], 1.0, 0.0)   
        # Take the difference of the signals in order to generate actual trading orders
        df['positions'] = df['signal'].diff()   
        df = df.drop(['signal'], axis=1)

        return df
    

    def plot_ticker_data(self, df):

#         df = df.rename(columns={'date':'index'}).set_index('index')

#         df = self.get_ticker_data(symbol)
        fig = plt.figure(figsize=(15,15))
        plt.plot(df.date, df.closePrice)
        plt.title(f'Symbol {symbol}')
        plt.tight_layout()
        
    