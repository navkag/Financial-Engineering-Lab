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
    stock_weekly_future = stock_weekly[stock_weekly.index > '2022-12-31']
    stock_weekly = stock_weekly[stock_weekly.index <= '2022-12-31']

    # Filter daily values from daily by averaging out prices during a month.
    stock_monthly = stock_daily.resample('M').first()
    stock_monthly_future = stock_monthly[stock_monthly.index > '2022-12-31']
    stock_monthly = stock_monthly[stock_monthly.index <= '2022-12-31']

    stock_daily_future = stock_daily[stock_daily.index > '2022-12-31']
    stock_daily = stock_daily[stock_daily.index <= '2022-12-31']

    # Calculate weekly log returns and weekly normalised returns.
    stock_weekly_returns = stock_weekly.pct_change().iloc[1:, :]
    stock_weekly_logreturns = np.log(stock_weekly_returns + 1)

    mu_weekly_return = stock_weekly_logreturns.mean() * 52
    sigma_weekly_return = stock_weekly_logreturns.std() * np.sqrt(52)

    mu_weekly_return += 0.5 * np.square(sigma_weekly_return)

    T_period = len(stock_weekly_future)
    delta_T = 1 / T_period

    S_0 = stock_weekly.iloc[-1, :]

    weekly_stock_prediction = [S_0]
    for i in range(T_period):
        Z = np.random.normal(0, 1, size=20)
        weekly_stock_prediction.append(weekly_stock_prediction[-1] * np.exp((mu_weekly_return - 0.5 * np.square(
            sigma_weekly_return)) * delta_T + sigma_weekly_return * np.sqrt(delta_T) * Z))

    weekly_stock_prediction = np.array(weekly_stock_prediction)

    # stock_weekly = (stock_weekly - mu_weekly_return) / sigma_weekly_return

    # x = np.log(stock_weekly.iloc[1:, :] / stock_weekly.iloc[:-1, :])
    # mean = np.mean(np.array(x))
    # std = np.std(np.array(x))
    # stock_weekly = (x - mean) / std

    # Calculate monthly log returns and monthly normalised returns.
    stock_monthly_returns = stock_monthly.pct_change().iloc[1:, :]
    stock_monthly_logreturns = np.log(stock_monthly_returns + 1)

    mu_monthly_return = stock_monthly_logreturns.mean() * 12
    sigma_monthly_return = stock_monthly_logreturns.std() * np.sqrt(12)

    mu_monthly_return += 0.5 * np.square(sigma_monthly_return)

    T_period = len(stock_monthly_future)
    delta_T = 1 / T_period

    S_0 = stock_monthly.iloc[-1, :]

    monthly_stock_prediction = [S_0]
    for i in range(T_period):
        Z = np.random.normal(0, 1, size=20)
        monthly_stock_prediction.append(monthly_stock_prediction[-1] * np.exp((mu_monthly_return - 0.5 * np.square(
            sigma_monthly_return)) * delta_T + sigma_monthly_return * np.sqrt(delta_T) * Z))

    monthly_stock_prediction = np.array(monthly_stock_prediction)

    # stock_monthly = (stock_monthly - mu_monthly_return) / \
    #     sigma_monthly_return

    # x = np.log(stock_monthly.iloc[1:, :] / stock_monthly.iloc[:-1, :])
    # mean = np.mean(np.array(x))
    # std = np.std(np.array(x))
    # stock_monthly = (x - mean) / std

    # Calculate daily log returns and daily normalised returns.
    stock_daily_returns = stock_daily.pct_change().iloc[1:, :]
    stock_daily_logreturns = np.log(stock_daily_returns + 1)

    mu_daily_return = stock_daily_logreturns.mean() * 252
    sigma_daily_return = stock_daily_logreturns.std() * np.sqrt(252)

    mu_daily_return += 0.5 * np.square(sigma_daily_return)

    T_period = len(stock_daily_future)
    delta_T = 1 / T_period

    S_0 = stock_daily.iloc[-1, :]

    daily_stock_prediction = [S_0]
    for i in range(T_period):
        Z = np.random.normal(0, 1, size=20)
        daily_stock_prediction.append(daily_stock_prediction[-1] * np.exp((mu_daily_return - 0.5 * np.square(
            sigma_daily_return)) * delta_T + sigma_daily_return * np.sqrt(delta_T) * Z))

    daily_stock_prediction = np.array(daily_stock_prediction)

    # stock_daily = (stock_daily - mu_daily_return) / sigma_daily_return

    # x = np.log(stock_daily.iloc[1:, :] / stock_daily.iloc[:-1, :])
    # mean = np.mean(np.array(x))
    # std = np.std(np.array(x))
    # stock_daily = (x - mean) / std

    x = np.linspace(-5, 5, 100)
    y = std_normal_dist(x, 0, 1)

    for i in range(20):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].plot(stock_daily.index, stock_daily.iloc[:, i],
                    label="Historical Data")
        axs[0].plot(stock_daily_future.index, daily_stock_prediction[1:, i],
                    label="Predicted Price")
        axs[0].plot(stock_daily_future.index, stock_daily_future.iloc[:, i],
                    label="Actual Price")
        axs[0].legend()
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].plot(stock_weekly.index, stock_weekly.iloc[:, i],
                    label="Historical Data")
        axs[1].plot(stock_weekly_future.index, weekly_stock_prediction[1:, i],
                    label="Predicted Price")
        axs[1].plot(stock_weekly_future.index, stock_weekly_future.iloc[:, i],
                    label="Actual Price")
        axs[1].legend()
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].plot(stock_monthly.index, stock_monthly.iloc[:, i],
                    label="Historical Data")
        axs[2].plot(stock_monthly_future.index, monthly_stock_prediction[1:, i],
                    label="Predicted Price")
        axs[2].plot(stock_monthly_future.index, stock_monthly_future.iloc[:, i],
                    label="Actual Price")
        axs[2].legend()
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(
            f'Predicted prices for {stock_names[stock_monthly.columns[i]]} stock from Jan 1st 2023')
        plt.tight_layout()
        plt.savefig(
            f'Q4stockplots/{stock_monthly.columns[i].replace(".", "_")}_predictions.png')
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
        index_weekly_future = index_weekly[index_weekly.index > '2022-12-31']
        index_weekly = index_weekly[index_weekly.index <= '2022-12-31']

        # Filter monthly values from daily by averaging out prices during a month.
        index_monthly = index_daily.resample('M').mean()
        index_monthly_future = index_monthly[index_monthly.index > '2022-12-31']
        index_monthly = index_monthly[index_monthly.index <= '2022-12-31']

        index_daily_future = index_daily[index_daily.index > '2022-12-31']
        index_daily = index_daily[index_daily.index <= '2022-12-31']

        # Calculate weekly log returns and weekly normalised returns.
        index_weekly_returns = index_weekly.pct_change().iloc[1:, :]
        index_weekly_logreturns = np.log(index_weekly_returns + 1)

        mu_weekly_return = index_weekly_logreturns.mean() * 52
        sigma_weekly_return = index_weekly_logreturns.std() * np.sqrt(52)

        mu_weekly_return += 0.5 * np.square(sigma_weekly_return)

        total_time = len(index_weekly_future)
        delta_T = 1 / total_time
        S_0 = index_weekly.iloc[-1, 0]

        predicted_prices_weekly = [S_0]

        for i in range(total_time):
            Z = np.random.normal(0, 1)
            predicted_prices_weekly.append(predicted_prices_weekly[-1] * np.exp(
                (mu_weekly_return - 0.5 * sigma_weekly_return ** 2) * delta_T + sigma_weekly_return * np.sqrt(delta_T) * Z))

        # index_weekly = (index_weekly - mu_weekly_return) / sigma_weekly_return

        # x = np.log(index_weekly.iloc[1:, :] / index_weekly.iloc[:-1, :])
        # mean = np.mean(np.array(x))
        # std = np.std(np.array(x))
        # index_weekly = (x - mean) / std

        # Calculate monthly log returns and monthly normalised returns.
        index_monthly_returns = index_monthly.pct_change().iloc[1:, :]
        index_monthly_logreturns = np.log(index_monthly_returns + 1)

        mu_monthly_return = index_monthly_logreturns.mean() * 12
        sigma_monthly_return = index_monthly_logreturns.std() * np.sqrt(12)

        mu_monthly_return += 0.5 * np.square(sigma_monthly_return)

        total_time = len(index_monthly_future)
        delta_T = 1 / total_time
        S_0 = index_monthly.iloc[-1, 0]

        predicted_prices_monthly = [S_0]

        for i in range(total_time):
            Z = np.random.normal(0, 1)
            predicted_prices_monthly.append(predicted_prices_monthly[-1] * np.exp(
                (mu_monthly_return - 0.5 * sigma_monthly_return ** 2) * delta_T + sigma_monthly_return * np.sqrt(delta_T) * Z))

        # index_monthly = (index_monthly - mu_monthly_return) / \
        #     sigma_monthly_return

        # x = np.log(index_monthly.iloc[1:, :] / index_monthly.iloc[:-1, :])
        # mean = np.mean(np.array(x))
        # std = np.std(np.array(x))
        # index_monthly = (x - mean) / std

        # Calculate daily log returns and daily normalised returns.
        index_daily_returns = index_daily.pct_change().iloc[1:, :]
        index_daily_logreturns = np.log(index_daily_returns + 1)

        mu_daily_return = index_daily_logreturns.mean() * 252
        sigma_daily_return = index_daily_logreturns.std() * np.sqrt(252)

        mu_daily_return += 0.5 * np.square(sigma_daily_return)

        total_time = len(index_daily_future)
        delta_T = 1 / total_time
        S_0 = index_daily.iloc[-1, 0]

        predicted_prices_daily = [S_0]

        for i in range(total_time):
            Z = np.random.normal(0, 1)
            predicted_prices_daily.append(predicted_prices_daily[-1] * np.exp(
                (mu_daily_return - 0.5 * sigma_daily_return ** 2) * delta_T + sigma_daily_return * np.sqrt(delta_T) * Z))

        # index_daily = (index_daily - mu_daily_return) / sigma_daily_return

        # x = np.log(index_daily.iloc[1:, :] / index_daily.iloc[:-1, :])
        # mean = np.mean(np.array(x))
        # std = np.std(np.array(x))
        # index_daily = (x - mean) / std

        x = np.linspace(-5, 5, 100)
        y = std_normal_dist(x, 0, 1)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        axs[0].plot(index_daily.index, index_daily.iloc[:, 0],
                    label="Historical Data")
        axs[0].plot(index_daily_future.index, index_daily_future.iloc[:, 0],
                    label="Actual Price")
        axs[0].plot(index_daily_future.index, predicted_prices_daily[1:],
                    label="Predicted Price")
        axs[0].legend()
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Price')
        axs[0].set_title('Daily')

        axs[1].plot(index_weekly.index, index_weekly.iloc[:, 0],
                    label="Historical Data")
        axs[1].plot(index_weekly_future.index, index_weekly_future.iloc[:, 0],
                    label="Actual Price")
        axs[1].plot(index_weekly_future.index, predicted_prices_weekly[1:],
                    label="Predicted Price")
        axs[1].legend()
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Price')
        axs[1].set_title('Weekly')

        axs[2].plot(index_monthly.index, index_monthly.iloc[:, 0],
                    label="Historical Data")
        axs[2].plot(index_monthly_future.index, index_monthly_future.iloc[:, 0],
                    label="Actual Price")
        axs[2].plot(index_monthly_future.index, predicted_prices_monthly[1:],
                    label="Predicted Price")
        axs[2].legend()
        axs[2].set_xlabel('Date')
        axs[2].set_ylabel('Price')
        axs[2].set_title('Monthly')

        plt.suptitle(f'Predicted prices for {index} from Jan 1 2023')
        plt.tight_layout()
        plt.savefig(f'Q4indexplots/{index.replace(" ", "_")}_predictions.png')
        # plt.show()
        plt.close()


def log_std_normal_dist(x, mean, std):
    exp_term = np.exp(-0.5 * np.square((np.log(x) - mean) / std))
    y = x * np.sqrt(np.pi * 2) * std

    return exp_term / y


def std_normal_dist(x, mean, std):
    x = np.exp(-0.5 * np.square((x - mean) / std))
    x /= np.sqrt(np.pi * 2) * std
    return x


def main():
    print('Creating Plots. Please wait...')

    folder_name = 'Q4indexplots'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    folder_name2 = 'Q4stockplots'
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
