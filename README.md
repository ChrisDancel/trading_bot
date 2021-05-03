# Trading Bot (Recommender)

This project demonstrates a simple trading advice pipeline that takes into account 30 da forward forecasts 
of company share prices and by extension, when certain simple moving average (SMA) indicator cross over. 
Potential buy and sell signals are collated for all companies and then emailed to a given email address with 
information on:
* stock name
* when key indicator is forecasted to cross

### Simple trading Strategy
* Buy signal: when SMA_50 positively crosses SMA_200
* Short signal: when SMA_10 negatively crosses SMA_50

### Intructions on running the code

1. Setup local database
    * example uses MYSQL with a named database called `stocks`
    
2. Setup API key with AMERITRADE
    * needed to get access to all tickers on NYSE
    * go to https://www.tdameritrade.com/home.html
    * open free account to get api token

3. Fill on config
    * rename `config.json.dist` to `config.json` in `config/` folder
    * fill in missing details
    
4. Pull and save historical share price data
    * run `python3 main.py -b`
    
5. Forecast signals
    * run `python3 main.py -f`
    * will persist buy and sell signals to file in `data/`
    
6. Email signals
    * run `python3 main -e`
    * will email signals
    
#### Disclaimer
This is my personal project aimed purely as a hobby to practice coding and to learn more about the stock market. 

I take no responsibility for anyone's trading activities based off the recommendations from this work. 

