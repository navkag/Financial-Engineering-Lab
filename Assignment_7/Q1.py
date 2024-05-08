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
    # Ques 1
    call_price, put_price = get_prices(100, 120, 0.05, 0.2, 0.5, 1)
    print(f"Spot: {100}, Strike: {120}, Risk-free rate: {5}%, Volatility: {20}%, Maturity time: {1}yr, Current time: {6}months.")
    print(f"Call option price: {call_price}")
    print(f"Put option price: {put_price}")


if __name__ == "__main__":
    main()
