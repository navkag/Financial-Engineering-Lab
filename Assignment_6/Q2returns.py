import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_stocks(filename, stock_names):
    stock_daily = pd.read_csv(filename)
    stock_daily['Date'] = pd.to_datetime(stock_daily['Date'])
    stock_daily.set_index("Date", inplace=True)

    # Filter weekly values from daily by averaging out prices during a week.
    stock_weekly = stock_daily.resample('W').first()

    # Filter daily values from daily by averaging out prices during a month.
    stock_monthly = stock_daily.resample('M').first()

    # Calculate weekly returns and weekly normalised returns.
    stock_weekly = stock_weekly.pct_change().iloc[1:, :]
    mu_weekly_return = stock_weekly.mean()
    sigma_weekly_return = stock_weekly.std()

    stock_weekly = (stock_weekly - mu_weekly_return) / sigma_weekly_return

    # Calculate monthly returns and monthly normalised returns.
    stock_monthly = stock_monthly.pct_change().iloc[1:, :]
    mu_monthly_return = stock_monthly.mean()
    sigma_monthly_return = stock_monthly.std()

    stock_monthly = (stock_monthly - mu_monthly_return) / \
        sigma_monthly_return

    # Calculate daily returns and daily normalised returns.
    stock_daily = stock_daily.pct_change().iloc[1:, :]
    mu_daily_return = stock_daily.mean()
    sigma_daily_return = stock_daily.std()

    stock_daily = (stock_daily - mu_daily_return) / sigma_daily_return
    x = np.linspace(-5, 5, 100)
    y = std_normal_dist(x, 0, 1)

    for i in range(20):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].hist(stock_daily.iloc[:, i],
                    label="Daily", bins=30, density=True)
        axs[0].plot(x, y, label="Std Normal dist")
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].hist(stock_weekly.iloc[:, i],
                    label="Weekly", bins=20, density=True)
        axs[1].plot(x, y, label="Std Normal dist")
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].hist(stock_monthly.iloc[:, i],
                    label="Monthly", bins=10, density=True)
        axs[2].plot(x, y, label="Std Normal dist")
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(
            f'Normalized return of {stock_names[stock_monthly.columns[i]]} stock.')
        plt.tight_layout()
        plt.savefig(
            f'Q2stockplots/{stock_monthly.columns[i].replace(".", "_")}_returns.png')
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

        # Calculate weekly returns and weekly normalised returns.
        index_weekly = index_weekly.pct_change().iloc[1:, :]
        mu_weekly_return = index_weekly.mean()
        sigma_weekly_return = index_weekly.std()

        index_weekly = (index_weekly - mu_weekly_return) / sigma_weekly_return

        # Calculate monthly returns and monthly normalised returns.
        index_monthly = index_monthly.pct_change().iloc[1:, :]
        mu_monthly_return = index_monthly.mean()
        sigma_monthly_return = index_monthly.std()

        index_monthly = (index_monthly - mu_monthly_return) / \
            sigma_monthly_return

        # Calculate daily returns and daily normalised returns.
        index_daily = index_daily.pct_change().iloc[1:, :]
        mu_daily_return = index_daily.mean()
        sigma_daily_return = index_daily.std()

        index_daily = (index_daily - mu_daily_return) / sigma_daily_return
        x = np.linspace(-5, 5, 100)
        y = std_normal_dist(x, 0, 1)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].hist(index_daily.iloc[:, 0],
                    label="Daily", bins=30, density=True)
        axs[0].plot(x, y, label="Std Normal dist")
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].hist(index_weekly.iloc[:, 0],
                    label="Weekly", bins=20, density=True)
        axs[1].plot(x, y, label="Std Normal dist")
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].hist(index_monthly.iloc[:, 0],
                    label="Monthly", bins=10, density=True)
        axs[2].plot(x, y, label="Std Normal dist")
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(f'Normalized return of {index}')
        plt.tight_layout()
        plt.savefig(f'Q2indexplots/{index.replace(" ", "_")}_returns.png')
        # plt.show()
        plt.close()


def std_normal_dist(x, mean, std):
    x = np.exp(-0.5 * np.square((x - mean) / std))
    x /= np.sqrt(np.pi * 2) * std
    return x


def main():
    print('Creating Plots. Please wait...')

    folder_name = 'Q2indexplots'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_name2 = 'Q2stockplots'
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
