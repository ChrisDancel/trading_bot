import logging
log = logging.getLogger(__name__)

import pandas as pd
import pytz
from sqlalchemy import create_engine
from datetime import datetime

today = datetime.today().astimezone(pytz.timezone("America/New_York"))
today_fmt = today.strftime("%Y-%m-%d")


def unix_time_millis(dt):
    # Function to turn a datetime object into unix
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000.0)


def unix_time_to_date_string(dt: int):
    result_ms = pd.to_datetime(dt, unit="ms")
    return datetime.strftime(result_ms, "%Y-%m-%d")


def date_string_to_unix_time(dt: str):
    return int(datetime.strptime(dt, "%Y-%m-%d").strftime("%s")) * 1000


def unix_timestamp_to_date(uts):
    # check if uts is in milliseconds
    if len(str(uts)) == 13:
        return datetime.utcfromtimestamp(uts / 1000).strftime("%Y-%m-%d")
    elif len(str(uts)) == 10:
        return datetime.utcfromtimestamp(uts).strftime("%Y-%m-%d")
    else:
        raise ValueError("unix timestamp input looks wrong: {}".format(uts))


def create_conn(**creds_json):
    engine = create_engine(
        "mysql+pymysql://{user}:{pw}@localhost/{db}".format(
            user=creds_json["username"],
            pw=creds_json["password"],
            db=creds_json["database"],
        )
    )

    return engine


def load_to_mysql(df, tablename, connection, if_exists="replace"):
    df.to_sql(con=connection, name=tablename, if_exists=if_exists)


def read_sql_to_df(sql: str, connection):
    return pd.read_sql(sql, connection)


def check_table_exists(conn, tablename):
    """
    check if a table exists in database
    """
    return True if tablename in conn.table_names() else False


def get_latest_unix_date(conn, tablename):
    """
    get latest timestamp in unix format from a particular table
    """
    return pd.read_sql(
        "select max(datetime_unix) from stocks.{}".format(tablename), conn
    ).values[0][0]
