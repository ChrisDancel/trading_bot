# Recommender Trading Bot

This project demonstrates a simple trading advice pipeline that takes into account 30 day forward forecasts 
of company share prices and by extension, when certain simple moving average (SMA) indicator cross over. 
Potential buy and sell signals are collated for all companies and then emailed to a given address with 
information on:
* stock name
* when key indicators are forecast to cross

### Simple trading Strategy
* Buy signal: when SMA_50 positively crosses SMA_200
* Short signal: when SMA_10 negatively crosses SMA_50

### Instructions on running the code

1. Setup local database
    * example uses MYSQL with a named database called `stocks`
    * [TODO] add integration to GCP
    
2. Setup API key with AMERITRADE
    * needed to get access to all tickers on NYSE
    * go to https://www.tdameritrade.com/home.html
    * open free account to get api token
    * [TODO] find data source api to LSE

3. Fill on config
    * rename `config.json.dist` to `config.json` in `config/` folder
    * fill in missing details
    
4. Pull and save historical share price data
    * run `python3 main.py -b`
    
5. Forecast for signals
    * run `python3 main.py -f`
    * will persist buy and sell signals to file in `data/`
    
6. Email signals
    * run `python3 main -e`
    * will email signals
    
7. Persist signals
    * run `python3 main -p`
    * persist buy/sell signals to table `recommendations`

### Automating process with Airflow

8. Setup airflow locally 
    * there are numerous websites to help setup airflow
    * currently this has been done on mac
    * once setup, copy `trading_bot_dag.py` into `dags/` folder
    * change project paths for `PROJECT_DIR` and `PYTHON_ENV`
    * run dag
    
Note:
   * PROJECT_DIR is the local directory that this project is located
   * PYTHON_ENV is the python path for the virtual environment of this project


#### Disclaimer
This is my personal project aimed purely as a hobby to practice coding and to learn more about the stock market. 

I take no responsibility for anyone's trading activities based off the recommendations from this work. 

