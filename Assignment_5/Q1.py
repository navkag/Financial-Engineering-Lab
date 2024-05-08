import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_slope_intercept(weights):
    n = len(weights)

    slopes = (weights[:n - 1, 1] - weights[1:, 1]) / \
        (weights[:n - 1, 0] - weights[1:, 0])
    slope = np.mean(slopes)

    intercepts = weights[1:, 1] - slopes * weights[1:, 0]
    intercept = np.mean(intercepts)

    return [slope, intercept]


def generate_minimum_variance_line(mu, C, target_return):
    u = np.ones_like(mu)
    C_inv = np.linalg.inv(C)

    e1 = u @ C_inv @ u.T
    e2 = u @ C_inv @ mu.T
    e3 = mu @ C_inv @ u.T
    e4 = mu @ C_inv @ mu.T

    m1 = np.array([[1, e2], [target_return, e4]])
    m2 = np.array([[e1, 1], [e3, target_return]])
    m3 = np.array([[e1, e2], [e3, e4]])

    weights = np.linalg.det(m1) * u @ C_inv + np.linalg.det(m2) * mu @ C_inv
    weights /= np.linalg.det(m3)

    return weights


def generate_minimum_variance_portfolio(mu, C):
    u = np.ones_like(mu)
    C_inv = np.linalg.inv(C)

    weights = u @ C_inv / (u @ C_inv @ u.T)
    return weights


def plot_markovitz_efficient_frontier(mu, C, name, title):
    sample_size = 1_000
    target_returns = np.linspace(0, 0.4, sample_size)

    min_var_port_weights = generate_minimum_variance_portfolio(mu, C)
    min_var_port_return = mu @ min_var_port_weights.T
    min_var_port_risk = np.sqrt(
        min_var_port_weights @ C @ min_var_port_weights.T)

    efficient_portfolios, non_efficient_portfolios = list(), list()
    efficient_weights = list()

    for i in range(sample_size):
        target_return = target_returns[i]
        target_weights = generate_minimum_variance_line(mu, C, target_return)
        target_return = mu @ target_weights.T
        target_risk = np.sqrt(target_weights @ C @ target_weights.T)

        if target_return >= min_var_port_return:
            efficient_portfolios.append(
                [target_return, target_risk])
            efficient_weights.append(target_weights)
        else:
            non_efficient_portfolios.append(
                [target_return, target_risk])

    efficient_portfolios = np.array(efficient_portfolios)
    non_efficient_portfolios = np.array(non_efficient_portfolios)

    plt.plot(
        efficient_portfolios[:, 1], efficient_portfolios[:, 0], label="Efficicent frontier", c="b")
    plt.plot(non_efficient_portfolios[:, 1],
             non_efficient_portfolios[:, 0], c="r")
    plt.scatter(min_var_port_risk, min_var_port_return,
                color="orange", label="Minimum risk portfolio")
    plt.text(min_var_port_risk + 0.005, min_var_port_return, f"({min_var_port_risk: .2f}, {min_var_port_return: .2f})",
             fontsize=12, verticalalignment='bottom')
    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.clf()


def plot_CML(mu, C, name, title, risk_free_return):
    u = np.ones_like(mu)
    C_inv = np.linalg.inv(C)
    sample_size = 1_000
    target_returns = np.linspace(0, 0.6, sample_size)

    min_var_port_weights = generate_minimum_variance_portfolio(mu, C)
    min_var_port_return = mu @ min_var_port_weights.T
    min_var_port_risk = np.sqrt(
        min_var_port_weights @ C @ min_var_port_weights.T)

    efficient_portfolios, non_efficient_portfolios = list(), list()
    efficient_weights = list()

    for i in range(sample_size):
        target_return = target_returns[i]
        target_weights = generate_minimum_variance_line(mu, C, target_return)
        target_return = mu @ target_weights.T
        target_risk = np.sqrt(target_weights @ C @ target_weights.T)

        if target_return >= min_var_port_return:
            efficient_portfolios.append(
                [target_return, target_risk])
            efficient_weights.append(target_weights)
        else:
            non_efficient_portfolios.append(
                [target_return, target_risk])

    efficient_portfolios = np.array(efficient_portfolios)
    non_efficient_portfolios = np.array(non_efficient_portfolios)

    market_weights = (mu - risk_free_return * u) @ C_inv
    market_weights /= np.sum(market_weights)

    market_return = market_weights @ mu.T
    market_risk = np.sqrt(market_weights @ C @ market_weights.T)

    print(
        f"\nMarket return: {market_return}\nMarket risk: {market_risk}\nMarket weights: {market_weights}")

    cml_risks = np.linspace(0, 0.6, 10_000)

    risk_premium = (market_return - risk_free_return) / market_risk

    print(
        f"\nCapital market line equation: return = {risk_free_return} + {risk_premium} * risk\n")

    cml_returns = risk_free_return + risk_premium * cml_risks

    plt.plot(
        efficient_portfolios[:, 1], efficient_portfolios[:, 0], label="Efficicent frontier", c="b")
    plt.plot(non_efficient_portfolios[:, 1],
             non_efficient_portfolios[:, 0], c="r")
    plt.plot(cml_risks, cml_returns,
             label="Capital market line", c="g")
    plt.scatter(market_risk, market_return,
                color="orange", label="Market portfolio")
    plt.text(market_risk + 0.005, market_return, f"({market_risk: .2f}, {market_return: .2f})",
             fontsize=12, verticalalignment='bottom')
    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    plt.clf()

    return [market_return, market_risk]


def plot_SML(market_return, risk_free_return, name, title, printstr):
    slope, intercept = market_return - risk_free_return, risk_free_return

    print(
        f"Security Market Line {printstr}: return = {slope} * beta + {intercept}.")

    beta = np.linspace(-0.5, 1.2, 30)
    returns = slope * beta + intercept

    plt.plot(beta, returns)
    plt.xlabel("Beta")
    plt.ylabel("Return")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(name)
    plt.show()


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

    # Get market returns for bse sensex 30 and nse nifty 50.
    sensex_mkt_return = bse_index.mean().iloc[0] * len(bse_index) / 5
    nifty_mkt_return = nse_index.mean().iloc[0] * len(bse_index) / 5

    # Define rate of return on risk free investment.
    risk_free_rate = 0.05

    mu_bse_stocks, C_bse_stocks = bse_stocks.mean() * len(bse_stocks) / \
        5, bse_stocks.cov() * len(bse_stocks) / 5
    mu_nse_stocks, C_nse_stocks = nse_stocks.mean() * len(nse_stocks) / \
        5, nse_stocks.cov() * len(nse_stocks) / 5
    mu_non_bse_stocks, C_non_bse_stocks = non_bse_stocks.mean() * len(non_bse_stocks) / \
        5, non_bse_stocks.cov() * len(non_bse_stocks) / 5
    mu_non_nse_stocks, C_non_nse_stocks = non_nse_stocks.mean() * len(non_nse_stocks) / \
        5, non_nse_stocks.cov() * len(non_nse_stocks) / 5

    plot_markovitz_efficient_frontier(
        mu_bse_stocks, C_bse_stocks, "BSE_stocks_markovitz_efficient_frontier.png", "Markovitz Efficient Frontier BSE Stocks")
    plot_markovitz_efficient_frontier(
        mu_nse_stocks, C_nse_stocks, "NSE_stocks_markovitz_efficient_frontier.png", "Markovitz Efficient Frontier NSE Stocks")
    plot_markovitz_efficient_frontier(
        mu_non_bse_stocks, C_non_bse_stocks, "Non_BSE_stocks_markovitz_efficient_frontier.png", "Markovitz Efficient Frontier Non BSE Stocks")
    plot_markovitz_efficient_frontier(
        mu_non_nse_stocks, C_non_nse_stocks, "Non_NSE_stocks_markovitz_efficient_frontier.png", "Markovitz Efficient Frontier Non NSE Stocks")

    # Market portfolio and Capital Market Line
    [BSE_return, BSE_risk] = plot_CML(
        mu_bse_stocks, C_bse_stocks, "BSE_stocks_CML.png", "Capital Market Line BSE Stocks", risk_free_rate)
    [NSE_return, NSE_risk] = plot_CML(
        mu_nse_stocks, C_nse_stocks, "NSE_stocks_CML.png", "Capital Market Line NSE Stocks", risk_free_rate)
    [non_BSE_return, non_BSE_risk] = plot_CML(
        mu_non_bse_stocks, C_non_bse_stocks, "Non_BSE_stocks_CML.png", "Capital Market Line Non BSE Stocks", risk_free_rate)
    [non_NSE_return, non_NSE_risk] = plot_CML(
        mu_non_nse_stocks, C_non_nse_stocks, "Non_NSE_stocks_CML.png", "Capital Market Line Non NSE Stocks", risk_free_rate)

    # Security Market Line
    plot_SML(sensex_mkt_return, risk_free_rate, "BSE_stocks_SML.png",
             "Security Market Line BSE Stocks", "BSE stocks")
    plot_SML(nifty_mkt_return, risk_free_rate, "NSE_stocks_SML.png",
             "Security Market Line NSE Stocks", "NSE stocks")
    plot_SML(non_BSE_return, risk_free_rate, "Non_BSE_stocks_SML.png",
             "Security Market Line Non BSE Stocks", "non BSE stocks")
    plot_SML(non_NSE_return, risk_free_rate, "Non_NSE_stocks_SML.png",
             "Security Market Line Non NSE Stocks", "non NSE stocks")


if __name__ == "__main__":
    main()
