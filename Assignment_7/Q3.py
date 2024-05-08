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
    # Ques 3
    T, K, r, sigma = 1, 1, 0.05, 0.6

    t = np.linspace(0, 1, 40)
    s = np.linspace(0.25, 2, 40)

    T_, S_ = np.meshgrid(t, s)

    Z_array1 = np.zeros_like(T_)
    Z_array2 = np.zeros_like(T_)

    for i in range(len(t)):
        for j in range(len(s)):
            cp, pp = get_prices(s[j], K, r, sigma, t[i], T)

            Z_array1[j, i] = cp
            Z_array2[j, i] = pp

    fig = plt.figure(figsize=(10, 4))

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_surface(T_, S_, Z_array1, cmap='viridis')

    ax.set_xlabel('Time')
    ax.set_ylabel('Spot ')
    ax.set_zlabel('Call Price')
    ax.set_title("Call price vs spot and time (continuous).")

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(T_, S_, Z_array2, cmap='viridis')

    ax.set_xlabel('Time')
    ax.set_ylabel('Spot ')
    ax.set_zlabel('Put Price')
    ax.set_title("Put price vs spot and time (continuous).")

    plt.savefig("Option price vs spot and time (continuous).")
    plt.clf()


if __name__ == "__main__":
    main()