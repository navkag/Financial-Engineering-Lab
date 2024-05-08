import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def get_prices(S, K, r, sigma, t, T):
    if t == T:
        return max(0, S - K), max(0, K - S)

    d1 = (np.log(S / K) + (T - t) * (r + (sigma ** 2) / 2)) / \
        (sigma * np.sqrt(T - t))
    d2 = d1 - (sigma * np.sqrt(T - t))

    C = S * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    P = K * np.exp(-r * (T - t)) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return C, P


def main():
    # Ques 4

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    # 1) k and r plot.

    k = np.linspace(0, 2, 40)
    r = np.linspace(0, 1, 40)

    T_, S_ = np.meshgrid(k, r)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(k)):
        for j in range(len(r)):
            cp, pp = get_prices(S, k[i], r[j], sigma, t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Risk-free-rate')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs strike and Risk-free-rate.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Risk-free-rate')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs strike and Risk-free-rate.")

    plt.savefig("Option price vs strike and Risk-free-rate.")
    plt.clf()

    # 2) k and sigma plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    k = np.linspace(0, 2, 40)
    sigma = np.linspace(0, 1, 40)

    T_, S_ = np.meshgrid(k, sigma)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(k)):
        for j in range(len(sigma)):
            cp, pp = get_prices(S, k[i], r, sigma[j], t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs strike and Volatility.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs strike and Volatility.")

    plt.savefig("Option price vs strike and Volatility.")
    plt.clf()

    # 3) k and maturity time plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    k = np.linspace(0, 2, 40)
    T = np.linspace(0, 5, 40)

    T_, S_ = np.meshgrid(k, T)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(k)):
        for j in range(len(T)):
            cp, pp = get_prices(S, k[i], r, sigma, t, T[j])

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs strike and Maturity Time.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs strike and Maturity Time.")

    plt.savefig("Option price vs strike and Maturity Time.")
    plt.clf()

    # 4) r and sigma plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    r = np.linspace(0, 1, 40)
    sigma = np.linspace(0, 1, 40)

    T_, S_ = np.meshgrid(r, sigma)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(r)):
        for j in range(len(sigma)):
            cp, pp = get_prices(S, K, r[i], sigma[j], t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Risk-free Rate and Volatility.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Risk-free Rate and Volatility.")

    plt.savefig("Option price vs Risk-free Rate and Volatility.")
    plt.clf()

    # 5) maturity time and risk free rate plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    r = np.linspace(0, 1, 40)
    T = np.linspace(0, 5, 40)

    T_, S_ = np.meshgrid(r, T)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(r)):
        for j in range(len(T)):
            cp, pp = get_prices(S, K, r[i], sigma, t, T[j])

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Risk-free Rate and Maturity Time.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Risk-free Rate and Maturity Time.")

    plt.savefig("Option price vs Risk-free Rate and Maturity Time.")
    plt.clf()

    # 6) maturity time and sigma plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    sigma = np.linspace(0, 1, 40)
    T = np.linspace(0, 5, 40)

    T_, S_ = np.meshgrid(sigma, T)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(sigma)):
        for j in range(len(T)):
            cp, pp = get_prices(S, K, r, sigma[i], t, T[j])

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Volatility and Maturity Time.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Maturity Time')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Volatility and Maturity Time.")

    plt.savefig("Option price vs Volatility and Maturity Time.")
    plt.clf()

    # 7) strike and spot plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    k = np.linspace(0, 2, 40)
    s = np.linspace(0, 2, 40)

    T_, S_ = np.meshgrid(k, s)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(k)):
        for j in range(len(s)):
            cp, pp = get_prices(s[j], k[i], r, sigma, t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Strike and Spot prices.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Strike and Spot prices.")

    plt.savefig("Option price vs Strike and Spot prices.")
    plt.clf()

    # 8) maturity time and spot plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    T = np.linspace(0, 5, 40)
    s = np.linspace(0, 2, 40)

    T_, S_ = np.meshgrid(T, s)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(T)):
        for j in range(len(s)):
            cp, pp = get_prices(s[j], K, r, sigma, t, T[i])

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Maturity Time')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Maturity Time and Spot prices.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Maturity Time')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Maturity Time and Spot prices.")

    plt.savefig("Option price vs Maturity Time and Spot prices.")
    plt.clf()

    # 9) spot and risk free rate plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    r = np.linspace(0, 1, 40)
    s = np.linspace(0, 2, 40)

    T_, S_ = np.meshgrid(r, s)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(r)):
        for j in range(len(s)):
            cp, pp = get_prices(s[j], K, r[i], sigma, t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Risk-free Rate and Spot prices.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Risk-free Rate')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Risk-free Rate and Spot prices.")

    plt.savefig("Option price vs Risk-free Rate and Spot prices.")
    plt.clf()

    # 10) spot and sigma plot.
    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    sigma = np.linspace(0, 1, 40)
    s = np.linspace(0, 2, 40)

    T_, S_ = np.meshgrid(sigma, s)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(sigma)):
        for j in range(len(s)):
            cp, pp = get_prices(s[j], K, r, sigma[i], t, T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs Volatility and Spot prices.")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Spot')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs Volatility and Spot prices.")

    # plt.tight_layout()

    plt.savefig("Option price vs Volatility and Spot prices.")
    plt.clf()

    # 11) price vs spot plot

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    S = np.linspace(0, 2, 40)
    call_price, put_price = [], []
    for s in S:
        cp, pp = get_prices(s, K, r, sigma, t, T)
        call_price.append(cp)
        put_price.append(pp)

    plt.plot(S, call_price, label='Call')
    plt.plot(S, put_price, label='Put')
    plt.xlabel('Spot price')
    plt.ylabel('Option price')
    plt.title('Option prices vs Spot price')
    plt.legend()

    plt.savefig(f"Option price vs spot price.")
    plt.clf()

    # 12) price vs strike plot

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    K = np.linspace(0, 2, 40)
    call_price, put_price = [], []
    for k in K:
        cp, pp = get_prices(S, k, r, sigma, t, T)
        call_price.append(cp)
        put_price.append(pp)

    plt.plot(K, call_price, label='Call')
    plt.plot(K, put_price, label='Put')
    plt.xlabel('Strike price')
    plt.ylabel('Option price')
    plt.title('Option prices vs Strike price')
    plt.legend()

    plt.savefig(f"Option price vs Strike price.")
    plt.clf()

    # 13) price vs maturity time plot

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    T_ = np.linspace(0, 5, 40)
    call_price, put_price = [], []
    for t_ in T_:
        cp, pp = get_prices(S, K, r, sigma, t, t_)
        call_price.append(cp)
        put_price.append(pp)

    plt.plot(T_, call_price, label='Call')
    plt.plot(T_, put_price, label='Put')
    plt.xlabel('Maturity Time')
    plt.ylabel('Option price')
    plt.title('Option prices vs Maturity Time')
    plt.legend()

    plt.savefig(f"Option price vs Maturity Time.")
    plt.clf()

    # 14) price vs sigma plot

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    sigma = np.linspace(0, 1, 40)
    call_price, put_price = [], []
    for sig in sigma:
        cp, pp = get_prices(S, K, r, sig, t, T)
        call_price.append(cp)
        put_price.append(pp)

    plt.plot(sigma, call_price, label='Call')
    plt.plot(sigma, put_price, label='Put')
    plt.xlabel('Volatility')
    plt.ylabel('Option price')
    plt.title('Option prices vs Volatility')
    plt.legend()

    plt.savefig(f"Option price vs Volatility.")
    plt.clf()

    # 15) price vs risk-free rate plot

    T, K, r, sigma, S, t = 1, 1, 0.05, 0.6, 1, 0

    R = np.linspace(0, 1, 40)
    call_price, put_price = [], []
    for r in R:
        cp, pp = get_prices(S, K, r, sigma, t, T)
        call_price.append(cp)
        put_price.append(pp)

    plt.plot(R, call_price, label='Call')
    plt.plot(R, put_price, label='Put')
    plt.xlabel('Risk-free Rate')
    plt.ylabel('Option price')
    plt.title('Option prices vs Risk-free Rate')
    plt.legend()

    plt.savefig(f"Option price vs Risk-free Rate.")
    plt.clf()


if __name__ == "__main__":
    main()
