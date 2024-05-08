import numpy as np
import math
import matplotlib.pyplot as plt
import random
import seaborn as sns

plt.style.use('seaborn-darkgrid')


def geometric_brownian_motion(S_initial, mu, sigma, steps):
    dt = 1.0 / 252
    W_t = np.random.randn(steps)
    prices = []

    for i in range(steps):
        S_t = S_initial * math.exp((mu - sigma ** 2)
                                   * dt + sigma * math.sqrt(dt) * W_t[i])
        prices.append(S_t)
        S_initial = S_t

    return prices


def reduce_variance(option_payoff, control_variate, rate, steps, dt):
    X_bar = np.mean(control_variate)
    Y_bar = np.mean(option_payoff)

    max_iter = len(option_payoff)
    num, denom = 0, 0

    for idx in range(max_iter):
        num += (control_variate[idx] - X_bar) * (option_payoff[idx] - Y_bar)
        denom += (control_variate[idx] - X_bar) * \
            (control_variate[idx] - X_bar)

    b = num / denom
    reduced_variate = []
    for idx in range(max_iter):
        reduced_variate.append(
            (option_payoff[idx] - b * (control_variate[idx] - X_bar) * math.exp(-rate * steps * dt)))

    return reduced_variate


def asian_option_pricing(S_initial, rate, sigma, strike, max_iter=1000, path_length=126, steps=126):
    dt = 1 / 252
    call_option_payoff, put_option_payoff = [], []
    control_variate_call, control_variate_put = [], []

    for i in range(max_iter):
        S = geometric_brownian_motion(S_initial, rate, sigma, path_length)
        V_call = max(np.mean(S) - strike, 0)
        V_put = max(strike - np.mean(S), 0)

        call_option_payoff.append(math.exp(-rate * steps * dt) * V_call)
        put_option_payoff.append(math.exp(-rate * steps * dt) * V_put)

        control_variate_call.append(
            math.exp(-rate * steps * dt) * max(strike - S[len(S) - 1], 0))
        control_variate_put.append(
            math.exp(-rate * steps * dt) * max(S[len(S) - 1] - strike, 0))

    call_option_payoff = reduce_variance(
        call_option_payoff, control_variate_call, rate, steps, dt)
    put_option_payoff = reduce_variance(
        put_option_payoff, control_variate_put, rate, steps, dt)

    return np.mean(call_option_payoff), np.mean(put_option_payoff), np.var(call_option_payoff), np.var(put_option_payoff)


def sensitivity_S_initial(rate, sigma, strike):
    S0 = np.linspace(70, 140, num=250)
    call, put = [], []

    for i in S0:
        call_price, put_price, _, _ = asian_option_pricing(
            i, rate, sigma, strike, 500, 150, 100)
        call.append(call_price)
        put.append(put_price)

    plt.plot(S0, call)
    plt.xlabel("Initial asset price (S0)")
    plt.ylabel("Asian call option price")
    plt.title("Asian Call Option Dependency on S0")
    plt.savefig("Asian_Call_Option_Dependency_on_S0.png")
    plt.close()

    plt.plot(S0, put)
    plt.xlabel("Initial asset price (S0)")
    plt.ylabel("Asian put option price")
    plt.title("Asian Put Option Dependency on S0")
    plt.savefig("Asian_Put_Option_Dependency_on_S0.png")
    plt.close()

    return call, put


def sensitivity_strike(S_initial, rate, sigma):
    strike = np.linspace(70, 140, num=250)
    call, put = [], []

    for i in strike:
        call_price, put_price, _, _ = asian_option_pricing(
            S_initial, rate, sigma, i, 500, 150, 100)
        call.append(call_price)
        put.append(put_price)

    plt.plot(strike, call)
    plt.xlabel("Strike price (K)")
    plt.ylabel("Asian call option price")
    plt.title("Asian Call Option Dependency on K")
    plt.savefig("Asian_Call_Option_Dependency_on_K.png")
    plt.close()

    plt.plot(strike, put)
    plt.xlabel("Strike price (K)")
    plt.ylabel("Asian put option price")
    plt.title("Asian Put Option Dependency on K")
    plt.savefig("Asian_Put_Option_Dependency_on_K.png")
    plt.close()

    return call, put


def sensitivity_rate(S_initial, sigma, strike):
    rate = np.linspace(0, 0.5, num=120, endpoint=False)
    call, put = [], []

    for i in rate:
        call_price, put_price, _, _ = asian_option_pricing(
            S_initial, i, sigma, strike, 500, 150, 100)
        call.append(call_price)
        put.append(put_price)

    plt.plot(rate, call)
    plt.xlabel("Risk-free rate (r)")
    plt.ylabel("Asian call option price")
    plt.title("Asian Call Option Dependency on r")
    plt.savefig("Asian_Call_Option_Dependency_on_r.png")
    plt.close()

    plt.plot(rate, put)
    plt.xlabel("Risk-free rate (r)")
    plt.ylabel("Asian put option price")
    plt.title("Asian Put Option Dependency on r")
    plt.savefig("Asian_Put_Option_Dependency_on_r.png")
    plt.close()

    return call, put


def sensitivity_sigma(S_initial, rate, strike):
    sigma = np.linspace(0, 1, num=120, endpoint=False)
    call, put = [], []

    for i in sigma:
        call_price, put_price, _, _ = asian_option_pricing(
            S_initial, rate, i, strike, 500, 150, 100)
        call.append(call_price)
        put.append(put_price)

    plt.plot(sigma, call)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Asian call option price")
    plt.title("Asian Call Option Dependency on sigma")
    plt.savefig("Asian_Call_Option_Dependency_on_sigma.png")
    plt.close()

    plt.plot(sigma, put)
    plt.xlabel("Volatility (sigma)")
    plt.ylabel("Asian put option price")
    plt.title("Asian Put Option Dependency on sigma")
    plt.savefig("Asian_Put_Option_Dependency_on_sigma.png")
    plt.close()

    return call, put


def main():
    for strike in [90, 105, 110]:
        call_price, put_price, call_var, put_var = asian_option_pricing(
            100, 0.05, 0.2, strike)
        print("\n\n************** For Strike = {} **************".format(strike))
        print("Asian call option price \t\t=", call_price)
        print("Variance in Asian call option price \t=", call_var)
        print()
        print("Asian put option price \t\t\t=", put_price)
        print("Variance in Asian put option price \t=", put_var)

    # Sensitivity Analysis
    sensitivity_S_initial(0.05, 0.2, 105)
    sensitivity_strike(100, 0.05, 0.2)
    sensitivity_rate(100, 0.2, 105)
    sensitivity_sigma(100, 0.05, 105)


if __name__ == "__main__":
    main()
