import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_stocks(filename, stock_names):
    stock_daily = pd.read_csv(filename)
    stock_daily['Date'] = pd.to_datetime(stock_daily['Date'])
    stock_daily.set_index("Date", inplace=True)

    # Filter weekly values from daily by taking price of one day in a week.
    stock_weekly = stock_daily.resample('W').first()

    # Filter daily values from daily by taking price of one day in a month.
    stock_monthly = stock_daily.resample('M').first()

    for i in range(20):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].plot(stock_daily.index,
                    stock_daily.iloc[:, i], label="Daily")
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].plot(stock_weekly.index,
                    stock_weekly.iloc[:, i], label="Weekly")
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].plot(stock_monthly.index,
                    stock_monthly.iloc[:, i], label="Monthly")
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(
            f'Prices of {stock_names[stock_monthly.columns[i]]} stock.')
        plt.tight_layout()
        plt.savefig(
            f'Q1stockplots/{stock_monthly.columns[i].replace(".", "_")}.png')
        # plt.show()
        plt.close()


def plot_indices(indices):
    for index in indices:
        filename = indices[index]
        index_daily = pd.read_csv(filename)
        index_daily['Date'] = pd.to_datetime(index_daily['Date'])
        index_daily.set_index("Date", inplace=True)

        # Filter weekly values from daily by averaging out prices during a week.
        index_weekly = index_daily.resample('W').mean()

        # Filter daily values from daily by averaging out prices during a month.
        index_monthly = index_daily.resample('M').mean()

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].plot(index_daily.index,
                    index_daily.iloc[:, 0], label="Daily")
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].plot(index_weekly.index,
                    index_weekly.iloc[:, 0], label="Weekly")
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].plot(index_monthly.index,
                    index_monthly.iloc[:, 0], label="Monthly")
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(f'Prices of {index}')
        plt.tight_layout()
        plt.savefig(f'Q1indexplots/{index.replace(" ", "_")}.png')
        # plt.show()
        plt.close()


def main():
    print('Creating Plots. Please wait...')

    folder_name = 'Q1indexplots'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_name2 = 'Q1stockplots'
    if not os.path.exists(folder_name2):
        os.makedirs(folder_name2)

    indices = {"NSE Nifty 50": "nifty50_data.csv",
               "BSE Sensex 30": "sensex_data.csv"}

    plot_indices(indices)

    nifty_50_stocks = {'ADANIENT.NS': "Adani Enterprises", 'ASIANPAINT.NS': "Asian Paints", 'AXISBANK.NS': "Axis Bank", 'BAJAJ-AUTO.NS': "Bajaj Automobiles",
                       'BPCL.NS': "Bharat Petroleum Corp. Ltd", 'BHARTIARTL.NS': "Airtel", 'BRITANNIA.NS': "Britannia", 'CIPLA.NS': "Cipla", 'HCLTECH.NS': "HCL", 'TATASTEEL.NS': "TATA Steel"}
    nifty_next_50_stocks = {'PEL.NS': "Piramal Enterprises", 'JUBLFOOD.NS': "Jubilant Foods", 'UBL.NS': "United Breweries", 'MRF.NS': "MRF", 'BERGEPAINT.NS': "Berger Paints",
                            'SIEMENS.NS': "Siemens", 'HINDPETRO.NS': "Hindustan Petroleum", 'HAVELLS.NS': "Havells", 'INDIGO.NS': "Indigo", 'DABUR.NS': "Dabur"}
    nsedata1_stocks = {**nifty_50_stocks, **nifty_next_50_stocks}

    sensex_30_stocks = {'SUNPHARMA.BO': "Sun Pharma", 'TITAN.BO': "Titan", 'WIPRO.BO': "Wipro", 'SBIN.BO': "State Bank of India",
                        'ULTRACEMCO.BO': "Ultra Cement", 'RELIANCE.BO': "Reliance Industries", 'MARUTI.BO': "Maruti Suzuki", 'LT.BO': "Larsen & Toubro", 'NTPC.BO': "NTPC Corp.", 'HINDUNILVR.BO': "Hindustan Unilever Ltd"}
    sensex_100_stocks = {'ZOMATO.BO': "Zomato", 'ZEEL.BO': "Zee Entertainment", 'TVSMOTOR.BO': "TVS Motors", 'PIDILITIND.BO': "Pidlite Indutries",
                         'NAUKRI.BO': "Naukri", 'IRCTC.BO': "IRCTC", 'DLF.BO': "DLF", 'BANKBARODA.BO': "Bank of Baroda", 'MFSL.BO': "Max Fin. Services", 'COLPAL.BO': "Colgate"}
    bsedata1_stocks = {**sensex_30_stocks, **sensex_100_stocks}

    plot_stocks("bsedata1.csv", bsedata1_stocks)
    plot_stocks("nsedata1.csv", nsedata1_stocks)

    print(f'Plots created. Check folders {folder_name} and {folder_name2}.')


if __name__ == "__main__":
    main()
