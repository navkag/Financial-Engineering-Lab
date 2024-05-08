import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def main():
    file_path = "US_stocks.csv"
    df = pd.read_csv(file_path, sep=",")
    df.set_index("Date", inplace=True)
    df = df.pct_change()

    mu = np.mean(df, axis=0).to_numpy() * 12
    C = df.cov().to_numpy() * 12

    u = np.ones_like(mu)
    C_inv = np.linalg.inv(C)

    sample_size = 10_000
    target_returns = np.linspace(0, 1, sample_size)

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

    plt.plot(efficient_portfolios[:, 1], efficient_portfolios[:,
                                                              0], label="Efficicent frontier", c="b")
    plt.plot(non_efficient_portfolios[:, 1],
             non_efficient_portfolios[:, 0], c="r")
    plt.scatter(min_var_port_risk, min_var_port_return,
                color="orange", label="Minimum risk portfolio")
    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Markowitz Efficient Frontier for 10 stocks.")

    plt.tight_layout()
    plt.savefig('markovitz_efficient_frontier_10_stocks.png')
    plt.show()
    plt.clf()

    # Market portfolio and Capital Market Line
    risk_free_return = 0.05

    market_weights = (mu - risk_free_return * u) @ C_inv
    market_weights /= np.sum(market_weights)

    market_return = market_weights @ mu.T
    market_risk = np.sqrt(market_weights @ C @ market_weights.T)

    print(
        f"\nMarket return: {market_return}\nMarket risk: {market_risk}\nMarket weights: {market_weights}")

    cml_risks = np.linspace(0, 0.6, 10_000)

    risk_premium = (market_return - risk_free_return) / market_risk

    print(
        f"\nCapital maket line equation: return = {risk_free_return} + {risk_premium} * risk\n")

    cml_returns = risk_free_return + risk_premium * cml_risks

    plt.plot(
        efficient_portfolios[:, 1], efficient_portfolios[:, 0], label="Efficicent frontier", c="b")
    plt.plot(non_efficient_portfolios[:, 1],
             non_efficient_portfolios[:, 0], c="r")
    plt.plot(cml_risks, cml_returns,
             label="Capital market line", c="g")
    plt.scatter(market_risk, market_return,
                color="orange", label="Market portfolio")
    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Capital Market Line for 10 Stocks and Risk-free asset.")

    plt.tight_layout()
    plt.savefig('capital_market_line_10_stocks.png')
    plt.show()

    # Security Market Line
    slope, intercept = market_return - risk_free_return, risk_free_return

    print(f"Security Market Line: return = {slope} * beta + {intercept}.")

    beta = np.linspace(-0.5, 1.2, 30)
    returns = slope * beta + intercept

    plt.plot(beta, returns)
    plt.xlabel("Beta")
    plt.ylabel("Return")
    plt.title("Security Market Line")
    plt.tight_layout()
    plt.savefig("Security_market_line_10_stocks.")
    plt.show()


if __name__ == "__main__":
    main()
