import logging
log = logging.getLogger(__name__)

import os
import pandas as pd
import numpy as np
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from numpy import datetime64
from pretty_html_table import build_table


class Mail:
    def __init__(
        self,
        from_email,
        to_email,
        from_pw,
        n_days_max,
        data_loc,
        df_buy_name,
        df_sell_name,
    ):
        self.from_email = from_email
        self.to_email = to_email
        self.from_pw = from_pw
        self.n_days_max = n_days_max
        self.data_loc = data_loc
        self.df_buy_name = df_buy_name
        self.df_sell_name = df_sell_name

    def send_mail(self, body, subject):
        message = MIMEMultipart()
        message["Subject"] = subject
        message["From"] = self.from_email
        message["To"] = self.to_email

        body_content = body
        message.attach(MIMEText(body_content, "html"))
        msg_body = message.as_string()

        server = SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(message["From"], self.from_pw)
        server.sendmail(message["From"], message["To"], msg_body)
        server.quit()

    def filter_for_specific_future(self, df: pd.DataFrame):

        if df.empty:
            return pd.DataFrame()
        else:
            current: datetime64 = np.datetime64("today")
            df["future"] = df["date"] - current
            df["day"] = df["date"].dt.day_name()

            str_n_days_max = str(self.n_days_max) + " days"
            df = df[(df["date"] > current) & (df["future"] < str_n_days_max)]
            df = df.sort_values(
                by=["future", "ticker"], ascending=[True, True]
            ).reset_index(drop=True)

            df_final = (
                df.groupby(["future", "date", "ticker"])[["closePrice"]]
                .max()
                .reset_index()
            )
            return df_final

    def get_email_data(self, df_path):
        df = pd.read_pickle(df_path)
        return self.filter_for_specific_future(df)

    def send_buy_email(self):
        data = self.get_email_data(os.path.join(self.data_loc, self.df_buy_name))

        if data.empty:
            log.info("no buy data to email")
            return

        output = build_table(data, "blue_light")
        self.send_mail(output, "Buy side stocks")
        return "Mail sent successfully."

    def send_sell_email(self):
        data = self.get_email_data(os.path.join(self.data_loc, self.df_sell_name))

        if data.shape[0] == 0:
            log.info("no sell data to email")
            return

        output = build_table(data, "blue_light")
        self.send_mail(output, "Sell side stocks")
        return "Mail sent successfully."
