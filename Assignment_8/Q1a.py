import numpy as np
from pandas import read_csv

from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def getHistoricalVolatility(data):
    # change = np.zeros(len(close) - 1)
    # for i in range(1, len(close)):
    #     change[i - 1] = (close[i] - close[i - 1]) / close[i - 1]
    change = data.pct_change()
    historicalVolatility = np.std(change) * (252 ** 0.5)
    return historicalVolatility


def model(market='BSE', index=False, num_fig=0):
    fields = ['Close']
    if index:
        if market == 'BSE':
            data = read_csv('sensex_data.csv')
        else:
            data = read_csv('nifty50_data.csv')
    else:
        if market == 'BSE':
            data = read_csv('bsedata1.csv')
            data = data.iloc[:, :-1]
        else:
            data = read_csv('nsedata1.csv')
            data = data.iloc[:, :-1]

    data.set_index('Date', inplace=True)

    if(index):
        print("\n" + market + 'Index')
    else:
        print('\nMarket : ' + market)
    ##################Q1####################
    lastMonth = data.iloc[len(data) - 21:]
    sig = getHistoricalVolatility(lastMonth)
    print("Historical Volatility \n", sig)


# stock_name = ['ADANIENT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
#               'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'HCLTECH.NS', 'TATASTEEL.NS']
# stock_name = stock_name +

market_name = ['BSE', 'NSE']

model('BSE', index=True)
model('NSE', index=True)


for market in market_name:
    model(market, num_fig=0)
