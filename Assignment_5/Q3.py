import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_market_portfolio(M, C, risk_free_rate):
    u = np.ones_like(M)
    C_inv = np.linalg.inv(C)

    weights = (M - risk_free_rate) @ C_inv
    weights /= np.sum(weights)

    return weights


def main():
    nifty_50_stocks = ['ADANIENT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
                       'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'HCLTECH.NS', 'TATASTEEL.NS']
    nifty_next_50_stocks = ['PEL.NS', 'JUBLFOOD.NS', 'UBL.NS', 'MRF.NS', 'BERGEPAINT.NS',
                            'SIEMENS.NS', 'HINDPETRO.NS', 'HAVELLS.NS', 'INDIGO.NS', 'DABUR.NS']

    sensex_30_stocks = ['SUNPHARMA.BO', 'TITAN.BO', 'WIPRO.BO', 'SBIN.BO',
                        'ULTRACEMCO.BO', 'RELIANCE.BO', 'MARUTI.BO', 'LT.BO', 'NTPC.BO', 'HINDUNILVR.BO']
    sensex_100_stocks = ['ZOMATO.BO', 'ZEEL.BO', 'TVSMOTOR.BO', 'PIDILITIND.BO',
                         'NAUKRI.BO', 'IRCTC.BO', 'DLF.BO', 'BANKBARODA.BO', 'MFSL.BO', 'COLPAL.BO']

    # Get stock data.
    bse_stocks = pd.read_csv('bsedata1.csv')
    nse_stocks = pd.read_csv('nsedata1.csv')
    bse_stocks.set_index('Date', inplace=True)
    nse_stocks.set_index('Date', inplace=True)

    # Calculate daily returns.
    non_bse_stocks = bse_stocks.iloc[:, 10:].pct_change().iloc[1:, :]
    bse_stocks = bse_stocks.iloc[:, :10].pct_change().iloc[1:, :]
    non_nse_stocks = nse_stocks.iloc[:, 10:].pct_change().iloc[1:, :]
    nse_stocks = nse_stocks.iloc[:, :10].pct_change().iloc[1:, :]

    # Get data from indices.
    nse_index = pd.read_csv('nifty50_data.csv')
    bse_index = pd.read_csv('sensex_data.csv')

    nse_index.set_index('Date', inplace=True)
    bse_index.set_index('Date', inplace=True)

    # Remove special trading days from indices.
    nse_index = nse_index.reindex(nse_stocks.index)
    bse_index = bse_index.reindex(bse_stocks.index)

    # Calculate daily returns.
    nse_index = nse_index.pct_change().iloc[1:, :]
    bse_index = bse_index.pct_change().iloc[1:, :]

    # Define rate of return on risk free investment.
    risk_free_rate = 0.05

    # Get beta for each stock.

    # 1. Betas for stocks in sensex 30.
    print("Stocks in BSE Sensex 30: ")
    print()
    bse_stock_and_index = pd.concat(
        [bse_stocks.iloc[:, :10], bse_index], axis=1)

    bse_stock_cov = bse_stock_and_index.cov()

    betas_sensex_30 = []
    for i in range(10):
        betas_sensex_30.append(
            bse_stock_cov.iloc[i, 10] / bse_stock_cov.iloc[10, 10])

    display_dict = {'Stock': sensex_30_stocks, 'Beta': betas_sensex_30}
    display_df = pd.DataFrame(display_dict)
    print(display_df)
    print()

    # print(betas_sensex_30)

    # 2. Betas for stocks in nifty 50.
    print("Stocks in NSE Nifty 50: ")
    print()
    nse_stock_and_index = pd.concat(
        [nse_stocks.iloc[:, :10], nse_index], axis=1)

    nse_stock_cov = nse_stock_and_index.cov()

    betas_nifty_50 = []
    for i in range(10):
        betas_nifty_50.append(
            nse_stock_cov.iloc[i, 10] / nse_stock_cov.iloc[10, 10])

    display_dict = {'Stock': nifty_50_stocks, 'Beta': betas_nifty_50}
    display_df = pd.DataFrame(display_dict)
    print(display_df)
    print()
    # print(betas_nifty_50)

    # 3. Betas for stocks not in sensex 30.
    print("Stocks not in BSE Sensex 30: ")
    print()
    non_bse_stock_and_index = pd.concat(
        [non_bse_stocks.iloc[:, :10], bse_index], axis=1)

    non_bse_stock_cov = non_bse_stock_and_index.cov()

    betas_sensex_100 = []
    for i in range(10):
        betas_sensex_100.append(
            non_bse_stock_cov.iloc[i, 10] / non_bse_stock_cov.iloc[10, 10])

    display_dict = {'Stock': sensex_100_stocks, 'Beta': betas_sensex_100}
    display_df = pd.DataFrame(display_dict)
    print(display_df)
    print()
    # print(betas_sensex_100)

    # 4. Betas for stocks not in nifty 50.
    print("Stocks not in NSE Nifty 50: ")
    print()
    non_nse_stock_and_index = pd.concat(
        [non_nse_stocks.iloc[:, :10], nse_index], axis=1)

    non_nse_stock_cov = non_nse_stock_and_index.cov()

    betas_nifty_next_50 = []
    for i in range(10):
        betas_nifty_next_50.append(
            non_nse_stock_cov.iloc[i, 10] / non_nse_stock_cov.iloc[10, 10])

    display_dict = {'Stock': nifty_next_50_stocks, 'Beta': betas_nifty_next_50}
    display_df = pd.DataFrame(display_dict)
    print(display_df)
    print()
    # print(betas_nifty_next_50)


if __name__ == "__main__":
    main()
