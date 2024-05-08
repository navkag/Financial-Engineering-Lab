import numpy as np
import matplotlib.pyplot as plt


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


def main():
    # Mean return matrix
    mu = np.array([0.1, 0.2, 0.15])
    # Dispersion (var-covariance of return)
    C = np.array([[0.005, -0.010, 0.004],
                  [-0.010, 0.040, -0.002],
                  [0.004, -0.002, 0.023]])

    sample_size = 1_000
    # target_returns = np.linspace(0.075, 0.225, sample_size)
    weights_15_risk, returns_15_risk = list(), list()

    min_var_port_weights = generate_minimum_variance_portfolio(mu, C)
    min_var_port_return = mu @ min_var_port_weights.T
    min_var_port_risk = np.sqrt(
        min_var_port_weights @ C @ min_var_port_weights.T)

    efficient_portfolios, non_efficient_portfolios = list(), list()
    efficient_weights = list()

    i = 0

    while i < sample_size:
        weights = np.random.rand(3)
        weights /= np.sum(weights)

        target_return = mu @ weights.T
        target_weights = generate_minimum_variance_line(mu, C, target_return)
        target_return = mu @ target_weights.T
        target_risk = np.sqrt(target_weights @ C @ target_weights.T)

        if np.sum(target_weights >= 0) != 3:
            continue

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

        i += 1

    efficient_portfolios = np.array(efficient_portfolios)
    non_efficient_portfolios = np.array(non_efficient_portfolios)

    # Plotting feasible region assuming no short sales.
    j = 0

    feasible_region = list()
    while j < 100_000:
        weights = np.random.rand(3)
        weights /= np.sum(weights)

        return_point = mu @ weights.T
        risk_point = np.sqrt(weights @ C @ weights.T)

        feasible_region.append([return_point, risk_point])

        j += 1

    feasible_region = np.array(feasible_region)

    # Plotting w1 and w2
    k = 0
    weights_12, weights_31, weights_23 = list(), list(), list()
    while k < 10_000:
        num1 = np.random.rand()
        num2 = np.random.rand()
        num_sum = num1 + num2

        num1 /= num_sum
        num2 /= num_sum

        weight_12, weight_31, weight_23 = np.array([
            num1, num2, 0]), np.array([num2, 0, num1]), np.array([0, num1, num2])

        return_point_12 = mu @ weight_12.T
        return_point_23 = mu @ weight_23.T
        return_point_31 = mu @ weight_31.T

        risk_point_12 = np.sqrt(weight_12 @ C @ weight_12.T)
        risk_point_31 = np.sqrt(weight_31 @ C @ weight_31.T)
        risk_point_23 = np.sqrt(weight_23 @ C @ weight_23.T)

        weights_12.append([return_point_12, risk_point_12])
        weights_31.append([return_point_31, risk_point_31])
        weights_23.append([return_point_23, risk_point_23])

        k += 1

    weights_12 = np.array(weights_12)
    weights_31 = np.array(weights_31)
    weights_23 = np.array(weights_23)

    plt.scatter(feasible_region[:, 1], feasible_region[:, 0],
                color="lightblue", alpha=0.3, label="Feasible region", s=2)
    plt.scatter(
        efficient_portfolios[:, 1], efficient_portfolios[:, 0], label="Efficicent frontier", c="b", s=2)
    # # plt.scatter(non_efficient_portfolios[:, 1],
    # # non_efficient_portfolios[:, 0], c="r", s=2)
    plt.scatter(min_var_port_risk, min_var_port_return,
                color="orange", label="Minimum risk portfolio")

    plt.scatter(weights_12[:, 1], weights_12[:, 0],
                color="purple", label="Assets 1 and 2", s=1)
    plt.scatter(weights_23[:, 1], weights_23[:, 0],
                color="violet", label="Assets 2 and 3", s=1)
    plt.scatter(weights_31[:, 1], weights_31[:, 0],
                color="pink", label="Assets 3 and 1", s=1)

    plt.legend()
    plt.xlabel("Risk")
    plt.ylabel("Return")
    plt.title("Minimum Variance Curves with no Short Selling.")

    plt.tight_layout()
    plt.savefig('Minimum_variance_curves.png')
    plt.show()
    plt.clf()

    # Plotting minimum variance curves on weights.
    returns = np.linspace(0.1, 0.2, 1_000)
    weights = list()

    for target_return in returns:
        weights.append(generate_minimum_variance_line(mu, C, target_return))

    weights = np.array(weights)

    weights_12, weights_23, weights_31 = weights[:, [
        0, 1]], weights[:, [1, 2]], weights[:, [2, 0]]

    weights_2_assets = [weights_12, weights_23, weights_31]

    for i in range(1, 4):
        [slope, intercept] = generate_slope_intercept(weights_2_assets[i - 1])
        print(
            f"Minimum portfolio line: w{i % 3 + 1} = {slope} * w{(i - 1) % 3 + 1} + {intercept}")
        x = np.linspace(-0.1, 1.1, 10)
        y = slope * x + intercept

        plt.plot([0, 0], [0, 1], label=f"w{(i - 1) % 3 + 1}=0")
        plt.plot([0, 1], [0, 0], label=f"w{i % 3 + 1}=0")
        plt.plot([0, 1], [1, 0], label=f"w{i % 3 + 1} + w{(i - 1) % 3 + 1}=1")

        plt.plot(x, y, label="minimum_variance_curve")

        plt.xlabel(f"w{(i - 1) % 3 + 1}")
        plt.ylabel(f"w{i % 3 + 1}")
        plt.title(f"Minimum Variance Curve w{i % 3 + 1} vs w{(i - 1) % 3 + 1}")
        plt.legend()

        plt.savefig(
            f"Minimum Variance Curve w{i % 3 + 1} vs w{(i - 1) % 3 + 1}")
        plt.tight_layout()
        plt.show()
        plt.clf()


if __name__ == "__main__":
    main()
