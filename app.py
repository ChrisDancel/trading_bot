import pandas as pd
import numpy as np
import json
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# from scipy.stats import norm
from utils import utils_trading_bot as utb

markdown_text = """
This application analyses individual stocks
"""

@st.cache
def get_data():
    return vis.get_all_data()

def main(vis, symbols):
    st.write('# Stock Analysis App')
    st.markdown(markdown_text)
    df_all = get_data()
    
    # initialise visualiser
    select_symbol = st.sidebar.selectbox(
        "select stock symbol",
        tuple(symbols)
    )
    
    f"you have chosen symbol {select_symbol}"
    df = vis.get_ticker_data_from_cache(df_all, select_symbol)

    # add sma selector
    select_sma = st.checkbox('sma')    
    
    if select_sma:
        df = vis.add_sma(df, 'closePrice', 'date')
        df = vis.add_crossover(df, short_col='ma_50', long_col='ma_200')
        
    df = df.rename(columns={'date':'index'}).set_index('index')
    
    # get dataframes of past buy and sell signals
    df_buy_hist = df.loc[df.positions == 1.0]
    df_sell_hist = df.loc[df.positions == -1.0]
    
    df = df.reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(20,10), nrows=1, ncols=1)
    plt.title('a title')
    
    if 'positions' in df.columns:
        plt.plot(df[df.columns.difference(['positions'])])
        # Plot the "buy" trades against AAPL
        ax1.plot(df.loc[df.positions == 1.0].index, 
                 df.ma_50[df.positions == 1.0],
                 '^', markersize=10, color='m')

        # Plot the "sell" trades against AAPL
        ax1.plot(df.loc[df.positions == -1.0].index, 
                 df.ma_50[df.positions == -1.0],
                 'v', markersize=10, color='k')
        ax1.legend(df.columns.difference(['positions']))
    else:
        plt.plot(df)        

    st.pyplot()
    
    "### Buy signals"
    st.dataframe(df_buy_hist)

    "### Sell signals"
    st.dataframe(df_sell_hist)
    
#     c = alt.Chart(df).mark_line().encode(
#         x=alt.X('date:Q', axis=alt.Axis(tickCount=df.shape[0], grid=False)),
#         y=alt.Y('closePrice:Q')
#     )
#     st.altair_chart(c, use_container_width=True)

if __name__ == '__main__':
    
    with open('configs/config.json') as f:
        config = json.load(f)
    
    conn = utb.create_conn(**config['MYSQL'])

    # initialise visualiser
    vis = utb.Visualiser(conn=conn)
    symbols = vis.get_all_symbols()['symbol'].to_list()
    
    main(vis, symbols)