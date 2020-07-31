import os
import logging
import argparse
import pickle
import json
import alpaca_trade_api as tradeapi
from datetime import date
from utils import utils_trading_bot as utb
from google.cloud import bigquery

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

log = logging.getLogger("trading bot")

BUCKET_NAME = 'trading-bot-data-bucket'
FILE_NAME = 'config.json'


def build_history(config):
    log.info('***** STARTING BUILDING HISTORICAL DATA *****')

    path_to_symbols = 'data/symbols_{}.data'.format(config['AMERITRADE']['exchange'])

    if os.path.exists(path_to_symbols):
        with open(path_to_symbols, 'rb') as filehandle:
            # read the data as binary data stream
            symbols_clean = pickle.load(filehandle)
    else:
        log.debug('getting symbols for stock exchange {}'.format(config['AMERITRADE']['exchange']))
        symbols_clean = utb.get_symbols(exchange=config['AMERITRADE']['exchange'])

        with open(path_to_symbols, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(symbols_clean, filehandle)
            log.debug('saving for stock exchange {} to {}'.format(config['AMERITRADE']['exchange'], path_to_symbols))

    stock_history_params = {'periodType': 'year',
                            'period': 3,
                            'frequencyType': 'daily',
                            'frequency': 1}

    save_option = 'replace'
    conn = utb.create_conn(**config['MYSQL'])

    # Checking that historical table exists
    hist_table_exists = utb.check_table_exists(conn=conn,
                                               tablename='historical_prices')
                         
        
    if hist_table_exists:
        log.info('historical data table exists already. Finding latest unix timestamp in table...')
        latest_unix_date = utb.get_latest_unix_date(conn, 'historical_prices')
    
        # Check if the latest date in GBQ is one day before today's date. If it is then we do not need to update the
        # table anymore
        today = date.today().strftime("%Y-%m-%d")
        latest_bigquery_date = utb.unix_timestamp_to_date(latest_unix_date)
        date_increment = 60*60*24*1000
        tomorrow_unix_date = latest_unix_date + date_increment
        latest_bigquery_date_plus_one_day = utb.unix_timestamp_to_date(tomorrow_unix_date)

        if today == latest_bigquery_date_plus_one_day:
            log.info('No need to update historical share prices - latest date is already last trading day {}!'.format(latest_bigquery_date))
            return
        else:
            log.info('Last date is {}; updating historical share prices...'.format(latest_bigquery_date))
            save_option = 'append'

        stock_history_params['startDate'] = tomorrow_unix_date

    log.debug('getting historical trading data for each quote...')
    log.debug('stock history params: \n{}'.format(stock_history_params))
    df = utb.clean_historical_quotes_data(symbols_chunked=symbols_clean,
                                          api_key=config['AMERITRADE']['consumer_key'],
                                          stock_history_params=stock_history_params)

    log.debug('shape of df before removing nans: {}'.format(df.shape))
    log.debug('removing nans...')
    df = df[df['closePrice'].notnull()].reset_index(drop=True)
    log.debug('shape of df after removing nans: {}'.format(df.shape))
    log.debug('df preview: \n{}'.format(df.head()))
    log.info('save option chosen: {}'.format(save_option))
    
#     log.info('adding SMA columns...')
#     df = utb.add_sma(df, col_groupby='symbol', 
#                     col_value='closePrice', 
#                     col_date='date', 
#                     sma_list=[5, 10, 20, 50, 100, 200])

#     log.info('adding crossover columns...')
#     df = utb.add_crossover_pairs(df)
    
    if config['DATA']['persist_historical_deltas']:
        log.debug('Loading data to msqyl...')    
        utb.load_to_mysql(df=df, 
                          tablename='historical_prices', 
                          connection=conn, 
                          if_exists=save_option)

    else:
        log.info('historical deltas not persisted')

    log.info('***** FINISHED BUILDING HISTORICAL DATA *****')


def buy_sell(config):
    log.info('***** STARTING PORTFOLIO OPTIMISATION *****')

    log.info('Loading in all historical stock prices')
    df = utb.get_historical_stockprices(config)

    log.info('getting current date..')
    sorted_dates = sorted(set(df['date']))
    current_data_date = sorted_dates[-1]
    log.info('current date set as {}'.format(current_data_date))

    log.info('computing momentum of each stock...')
    df['momentum'] = df.groupby('symbol')['close'].rolling(
        config['MOMENTUM']['momentum_window'],
        min_periods=config['MOMENTUM']['minimum_momentum']
    ).apply(utb.momentum_score).reset_index(level=0, drop=True)

    # Current portfolio value
    portfolio_value, df_pf = utb.calc_portfolio_value(config)

    if portfolio_value > 0:
        portfolio_value = portfolio_value
    else:
        portfolio_value = config['PORTFOLIO']['portfolio_value']

    log.info('computing momentum stocks to buy...')
    df_buy = utb.get_momentum_stocks(
        df=df,
        date=current_data_date,
        portfolio_size=config['PORTFOLIO']['portfolio_size'],
        cash=portfolio_value
    )

    log.info('computing stocks to buy to optimise portfolio...')
    df_buy_new = utb.buy_new_stock(
        df_pf=df_pf,
        df_buy=df_buy
    )

    log.info('new stocks to purchase: \n{}'.format(df_buy_new))

    log.info('Calculating stocks to sell...')
    # Create a list of stocks to sell based on what is currently in our pf
    sell_list = list(set(df_pf['symbol'].tolist()) - set(df_buy['symbol'].tolist()))

    df_sell = utb.sell_stocks(
        df=df,
        df_pf=df_pf,
        sell_list=sell_list,
        date=current_data_date
    )

    df_sell_final = utb.stock_diffs(
        df_sell=df_sell,
        df_pf=df_pf,
        df_buy=df_buy
    )
    log.info(' stocks to sell: \n{}'.format(df_sell_final))

    api = tradeapi.REST(
        config['ALPACA']['alpaca_api_key'],
        config['ALPACA']['alpaca_secret_key'],
        config['ALPACA']['base_url'],
        'v2'
    )

    if df_buy_new is not None:
        if config['ALPACA']['auto_trade']:
            utb.order_stock(df=df_buy_new,
                            api=api,
                            side='buy')
            log.debug('buy order sent')
        else:
            buy_choice = input("do you want to confirm buy (yes/no)?")
            if buy_choice == 'yes':
                utb.order_stock(df=df_buy_new,
                                api=api,
                                side='buy')
                log.debug('buy order sent')
    else:
        log.debug('No stocks to purchase')

    if df_sell_final is not None:
        if config['ALPACA']['auto_trade']:
            utb.order_stock(df=df_sell_final,
                            api=api,
                            side='sell')
            log.debug('sell order sent')
        else:
            sell_choice = input("do you want to confirm sell (yes/no)?")
            if sell_choice == 'yes':
                utb.order_stock(df=df_sell_final,
                                api=api,
                                side='sell')
                log.debug('sell order sent')
    else:
        log.debug('No stocks to sell')

    log.info('***** FINISHED OPTIMISING PORTFOLIO *****')


def main(config):
    if args.purpose == 'build_history':
        build_history(config)
    elif args.purpose == 'buy_sell':
        buy_sell(config)
    else:
        raise ValueError('something went wrong with selection or code...')


if __name__ == '__main__':
    # config = utb.get_config_from_gcp(bucket_name=BUCKET_NAME,
    #                                  file_name=FILE_NAME)

    with open('configs/config.json') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--purpose',
                        choices={'build_history',
                                 'buy_sell'},
                        help='What do you want the model to do?',
                        required=True)
    args = parser.parse_args()
    main(config)
    