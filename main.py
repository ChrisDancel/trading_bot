import logging
import sys
import argparse
import json
from utils import utils_data_build as udb
from utils import utils_data_forecaster as udf
from utils import utils_email as ue
from utils import utils_shared as ush

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
)

log = logging.getLogger("trading bot")

FILE_NAME = "config.json"
DATA_DIR = "data/"


def build_history(config):
    log.info("***** STARTING BUILDING HISTORICAL DATA *****")

    log.debug("Getting fresh ticker data...")
    symbols = udb.Ticker.get_symbols()
    symbols_clean = list(symbols.keys())[0:3]

    log.info("Generating Data Store object...")
    conn = ush.create_conn(**config["MYSQL"])
    tablename_historical_prices = config["MYSQL"]["tablename_historical_data"]
    data_store = udb.DataStore(conn, tablename_historical_prices)

    log.info("Generating Data Source object...")
    data_source = udb.DataSource(api_key=config["AMERITRADE"]["consumer_key"])

    log.info("Generating Data Maintenance object...")
    data_maintenance = udb.DataMaintenance(
        data_store=data_store, data_source=data_source, ticker_list=symbols_clean
    )

    data_maintenance.update_all()

    log.info("***** FINISHED BUILDING HISTORICAL DATA *****")


def forecast(config):
    log.info("***** GENERATING FORECASTS... *****")

    log.info("Generating Data Store object...")
    conn = ush.create_conn(**config["MYSQL"])
    tablename_historical_prices = config["MYSQL"]["tablename_historical_data"]
    data_store = udb.DataStore(conn, tablename_historical_prices)

    log.info("Initialising Forecaster...")
    fc = udf.Forecaster(data_store=data_store, data_loc=DATA_DIR)

    log.debug("Loading all stock data...")
    df_all = fc.get_all_data()

    log.debug("Loading all symbols data..")
    symbols = fc.get_all_symbols()

    log.debug("Forecasting for all symbols...")
    df_buy_hist, df_sell_hist = fc.collect_all_future_signals(
        df=df_all,
        symbols=symbols,
        days_to_forecast=config["FORECAST"]["days_into_future"],
    )

    log.info("***** COMPLETED FORECASTS *****")

    summary = {
        "stage": "forecast",
        "df_buy_hist": df_buy_hist.shape,
        "df_sell_hist": df_sell_hist.shape,
    }

    return summary


def email(config):
    """
    Note - config = config['EAMAI']
    Args:
        config:

    Returns:

    """
    log.info("***** EMAILING FUTURE SIGNALS.. *****")

    n_days_max = config["n_days_max"]
    mail = ue.Mail(
        from_email=config["from_email"],
        to_email=config["to_email"],
        from_pw=config["from_pw"],
        n_days_max=n_days_max,
        data_loc=DATA_DIR,
        df_buy_name="df_buy_hist",
        df_sell_name="df_sell_hist",
    )

    mail.send_buy_email()
    mail.send_sell_email()

    summary = {"stage": "emails", "n_days_max": n_days_max}

    log.info("***** EMAILS SENT *****")

    return summary


def parse_args(sys_args: list) -> argparse.Namespace:
    """Parse args. Separate function to enable unit testing.

    Parameters
    ----------
    sys_args: iterable, sys.argv[1:]

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="trading bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build_history",
        "-b",
        action="store_true",
        help="Build and update share history",
    )
    parser.add_argument(
        "--forecast", "-f", action="store_true", help="Build time series forecasts"
    )
    parser.add_argument("--email", "-e", action="store_true", help="Email results")

    return parser.parse_args(sys_args)


def main(args: argparse.Namespace, config=dict):
    log.debug("".format(args))

    if args.build_history:
        build_history(config)

    if args.forecast:
        forecast(config)

    if args.email:
        email(config["EMAIL"])


if __name__ == "__main__":
    with open("configs/config.json") as f:
        config = json.load(f)

    args = parse_args(sys.argv[1:])
    main(args, config)
