import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    # Mean return matrix
    mu = np.array([0.1, 0.2, 0.15])
    u = np.ones_like(mu)
    # Dispersion (var-covariance of return)
    C = np.array([[0.005, -0.010, 0.004],
                  [-0.010, 0.040, -0.002],
                  [0.004, -0.002, 0.023]])
    C_inv = np.linalg.inv(C)

    sample_size = 1_000
    target_returns = np.linspace(0, 0.3, sample_size)
    weights_15_risk, returns_15_risk = list(), list()

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

        if abs(target_risk - 0.15) < 3e-4:
            weights_15_risk.append(target_weights)
            returns_15_risk.append(target_return)

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
    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Markowitz Efficient Frontier")

    plt.tight_layout()
    plt.savefig('markovitz_efficient_frontier.png')
    plt.show()
    plt.clf()

    # 10 different samples on the efficient frontier.
    idx = np.round(np.linspace(
        0, len(efficient_portfolios) - 1, 10)).astype(int)

    sample_weights, sample_returns, sample_risks = list(), list(), list()

    for index, i in enumerate(idx):
        sample_weights.append(efficient_weights[i])
        sample_returns.append(efficient_portfolios[i, 0])
        sample_risks.append(efficient_portfolios[i, 1])

    sample_weights = np.array(sample_weights)

    df = pd.DataFrame({"Weight 1": sample_weights[:, 0], "Weight 2": sample_weights[:, 1], "Weight 3": sample_weights[:, 2],
                       "Returns": sample_returns, "Risks": sample_risks})
    df.index = np.arange(1, len(df) + 1)
    print(df)

    print()

    # max and min return for 15% risk.
    print(
        f"Maximum return for portfolio with 15% risk: {returns_15_risk[1]}\nWeights: {weights_15_risk[1]}")
    print(
        f"Minimum return for portfolio with 15% risk: {returns_15_risk[0]}\nWeights: {weights_15_risk[0]}")

    # minimum risk portfolio for 18% return.
    target_weights = generate_minimum_variance_line(mu, C, target_return=0.18)
    target_return = mu @ target_weights.T
    target_risk = np.sqrt(target_weights @ C @ target_weights.T)

    print(
        f"\nMinimum risk portfolio for 18% return has risk: {target_risk}\nWeights: {target_weights}")

    # Market portfolio and Capital Market Line
    risk_free_return = 0.1

    market_weights = (mu - risk_free_return * u) @ C_inv
    market_weights /= np.sum(market_weights)

    market_return = market_weights @ mu.T
    market_risk = np.sqrt(market_weights @ C @ market_weights.T)

    print(
        f"\nMarket return: {market_return}\nMarket risk: {market_risk}\nMarket weights: {market_weights}")

    cml_risks = np.linspace(0, 0.3, 100)

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
    plt.title("Capital Market Line")

    plt.tight_layout()
    plt.savefig('capital_market_line.png')
    plt.show()

    # Get mixed portfolios having 10% and 25% risk.
    return_market_risk = risk_free_return + \
        (market_return - risk_free_return) * 0.25 / market_risk
    risk_free_asset_weight = (
        return_market_risk - market_return) / (risk_free_return - market_return)
    market_portfolio_weights = (1 - risk_free_asset_weight) * market_weights

    print(
        f"\nPortfolio having both risk and risk-free asset with 25% risk:\nRisk free asset weight: {risk_free_asset_weight}\nRisky asset weights: {market_portfolio_weights}")

    return_market_risk = risk_free_return + \
        (market_return - risk_free_return) * 0.1 / market_risk
    risk_free_asset_weight = (
        return_market_risk - market_return) / (risk_free_return - market_return)
    market_portfolio_weights = (1 - risk_free_asset_weight) * market_weights

    print(
        f"\nPortfolio having both risk and risk-free asset with 10% risk:\nRisk free asset weight: {risk_free_asset_weight}\nRisky asset weights: {market_portfolio_weights}")


if __name__ == "__main__":
    main()
