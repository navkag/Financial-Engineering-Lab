import yfinance as yf
import pandas as pd

# Get data for 10 stocks listed on NSE NIFTY and another 10 not listed on that index.
'''
Reliance Industries, Tata Consultancy Services, HDFC Bank, ICICI Bank, Infosys,
Hindustan Unilever, ITC, State Bank of India, LIC India, Bharti Airtel.
'''
# stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
#           'INFY.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'LICI.NS', 'BHARTIARTL.NS',
#           "SBUX", "BA", "ORCL", "GS", "WMT",
#           "F", "COST", "BRK-A", "XOM", "DIS"]

nifty_50_stocks = ['ADANIENT.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
                   'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'HCLTECH.NS', 'TATASTEEL.NS']
nifty_next_50_stocks = ['PEL.NS', 'JUBLFOOD.NS', 'UBL.NS', 'MRF.NS', 'BERGEPAINT.NS',
                        'SIEMENS.NS', 'HINDPETRO.NS', 'HAVELLS.NS', 'INDIGO.NS', 'DABUR.NS']

nifty_stocks = nifty_50_stocks + nifty_next_50_stocks

# Download historical data for each stock
data = yf.download(nifty_stocks, start='2019-01-01',
                   end='2023-12-31')['Adj Close']

# Save the data to a CSV file
data.to_csv('nsedata1.csv')


# Get data for stocks 10 listed on BSE SENSEX and another 10 not listed on BSE SENSEX.
'''
Asian Paints, Axis Bank, Bajaj Auto, Bajaj Finance (consumer), Bajaj Finserv (insurance),
Dr Reddy's, HCL Technologies, JSW Steel, KOTAK Mahindra Bank, Maruti.
'''
# stocks = ['ASIANPAINT.BO', 'AXISBANK.BO', 'BAJAJ-AUTO.BO', 'BAJFINANCE.BO',
#           'BAJAJFINSV.BO', 'DRREDDY.BO', 'HCLTECH.BO', 'JSWSTEEL.BO', 'KOTAKBANK.BO', 'MARUTI.BO',
#           "AAPL", "GOOGL", "MSFT", "AMZN", "META",
#           "TSLA", "NFLX", "NVDA", "ADBE", "PYPL"]

sensex_30_stocks = ['SUNPHARMA.BO', 'TITAN.BO', 'WIPRO.BO', 'SBIN.BO',
                    'ULTRACEMCO.BO', 'RELIANCE.BO', 'MARUTI.BO', 'LT.BO', 'NTPC.BO', 'HINDUNILVR.BO']
sensex_100_stocks = ['ZOMATO.BO', 'ZEEL.BO', 'TVSMOTOR.BO', 'PIDILITIND.BO',
                     'NAUKRI.BO', 'IRCTC.BO', 'DLF.BO', 'BANKBARODA.BO', 'MFSL.BO', 'COLPAL.BO']


sensex_stocks = sensex_30_stocks + sensex_100_stocks

# Download historical data for each stock
data = yf.download(sensex_stocks, start='2019-01-01',
                   end='2023-12-31')['Adj Close']

# Save the data to a CSV file
data.to_csv('bsedata1.csv')


# NSE Nifty50 data.
nifty_data = yf.download('^NSEI', start='2019-01-01',
                         end='2023-12-31')['Adj Close']

# Save data to CSV file
nifty_data.to_csv('nifty50_data.csv')


# BSE Sensex data.
bse_sensex = yf.download("^BSESN", start="2019-01-01",
                         end="2023-12-31")['Adj Close']

# Save the data as a CSV file
bse_sensex.to_csv("sensex_data.csv")
