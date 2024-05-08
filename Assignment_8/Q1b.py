import numpy as np
from pandas import read_csv
from tabulate import tabulate
from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def getS0(name='', market='BSE', index=False):
    fields = ['Close']
    if index:
        data = read_csv('./Data/' + market.lower() +
                        'data1.csv', usecols=fields, index_col=False)
    else:
        data = read_csv('./Data/' + market + 'Data/' + name +
                        '.csv', usecols=fields, index_col=False)
    return data.iloc[-1]['Close']


def getCall(S, K, r, t, sig):
    d1 = (np.log(S / K) + t * (r + (sig**2) / 2)) / (sig * (t**0.5))
    d2 = d1 - sig * (t**0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    C = S * Nd1 - K * np.exp(-r * t) * Nd2
    return C


def getPut(S, K, r, t, sig):
    d1 = (np.log(S / K) + t * (r + (sig**2) / 2)) / (sig * (t**0.5))
    d2 = d1 - sig * (t**0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    P = K * np.exp(-r * t) * Nd2 - S * Nd1
    return P


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
            print('BSE sensex index\n')
            data = read_csv('sensex_data.csv')
        else:
            print('NSE nifty index\n')
            data = read_csv('nifty50_data.csv')
    else:
        if market == 'BSE':
            data = read_csv('bsedata1.csv')
            data = data.iloc[:, :-1]
        else:
            data = read_csv('nsedata1.csv')
            data = data.iloc[:, :-1]

    data.set_index('Date', inplace=True)

    stock_names = data.columns

    lastMonth = data.iloc[len(data) - 21:]
    sig = getHistoricalVolatility(lastMonth)
    ##################Q2####################
    S0 = data.iloc[-1, :]
    A = np.arange(0.5, 1.6, 0.1)
    K = S0
    r = 0.05
    t = 126 / 252

    callPrice = list()
    putPrice = list()

    for a in A:
        callPrice.append(getCall(S0, a * K, r, t, sig))
        putPrice.append(getPut(S0, a * K, r, t, sig))

    callPrice = np.array(callPrice)
    putPrice = np.array(putPrice)

    callPrice = callPrice.T
    putPrice = putPrice.T

    if index:
        # print('Call Price = ', end='')
        # print(np.round(callPrice, 2))
        # print('Put Price = ', end='')
        # print(np.round(putPrice, 2))
        # print('\n')

        data = list(zip(A, callPrice[0], putPrice[0]))
        headers = ["A", "Call Price", "Put Price"]
        table = tabulate(data, headers=headers, tablefmt="grid")

        print(table)
        print()

    else:
        # print(callPrice.shape)
        for i, stock_name in enumerate(stock_names):
            print(stock_name + "\n")
            # print('Call Price = ', end='')
            # print(np.round(callPrice[i, :], 2))
            # print('Put Price = ', end='')
            # print(np.round(putPrice[i, :], 2))
            # print('\n')
            data = list(zip(A, callPrice[i, :], putPrice[i, :]))
            headers = ["A", "Call Price", "Put Price"]
            table = tabulate(data, headers=headers, tablefmt="grid")

            print(table)
            print()


market_name = ['BSE', 'NSE']

model('BSE', index=True)
model('NSE', index=True)


for market in market_name:
    model(market, num_fig=0)
