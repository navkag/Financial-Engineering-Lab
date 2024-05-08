import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M_list_1 = [i for i in range(1, 252, 1)]
M_list_5 = [i for i in range(1, 256, 5)]
S_0, K, T, r, sigma = 100, 105, 5, 0.05, 0.4
C_prices_1, P_prices_1 = list(), list()
C_prices_5, P_prices_5 = list(), list()
cnt_1, cnt_5 = 0, 0

print("Running computations with M ranging from 1 to 250 with step size 1.")
for M in M_list_1:
    delta_T = T / M
    u = np.exp(sigma * np.sqrt(delta_T) + (r - 0.5 * np.square(sigma)) * delta_T)
    d = np.exp(-sigma * np.sqrt(delta_T) + (r - 0.5 * np.square(sigma)) * delta_T)
    p_tilda = (np.exp(r * delta_T) - d) / (u - d)
    q_tilda = 1 - p_tilda

    # To check whether no-arbitrage condition is satisfied we need to check whether: 0 < d < e^r < u holds. (here r is
    # the interest rate for the duration of one time step)
    condition1 = 0 < d
    condition2 = d < np.exp(r * delta_T)
    condition3 = np.exp(r * delta_T) < u

    if condition1 and condition2 and condition3:
        cnt_1 += 1 # Keep track of all configs for which no-arbitrage is satisfied.
    else:
        print(f"For M = {M}, \tno-arbitrage condition not satisfied.")
        break   # Or stop computing for that config.

    C_list, P_list = list(), list()

    for i in range(M + 1):
        S_price = S_0 * (u ** (M - i)) * (d ** i)
        C_price = max(0, S_price - K)
        P_price = max(0, K - S_price)
        C_list.append(C_price)
        P_list.append(P_price)

    for i in range(M):
        C_list_new, P_list_new = list(), list()
        for j in range(len(C_list) - 1):
            C_new = np.exp(-r * delta_T) * (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
            C_list_new.append(C_new)

            P_new = np.exp(-r * delta_T) * (P_list[j] * p_tilda + P_list[j + 1] * q_tilda)
            P_list_new.append(P_new)
        P_list, C_list = P_list_new, C_list_new

    C_prices_1.append(C_list[0])
    P_prices_1.append(P_list[0])

prices_1 = {"M": M_list_1,
          "Call": C_prices_1,
          "Put": P_prices_1}

if cnt_1 == len(M_list_1):
    print("No-arbitrage condition satisfied for all configurations.\n")
else:
    print("The above mentioned configurations failed no-arbitrage condition.\n")

print(f"Call_option price:  {prices_1['Call'][-1]}")
print(f"Put option price: {prices_1['Put'][-1]}\n\n")

print("Running computations with M ranging from 1 to 250 with step size 5.")

for M in M_list_5:
    delta_T = T / M
    u = np.exp(sigma * np.sqrt(delta_T) + (r - 0.5 * np.square(sigma)) * delta_T)
    d = np.exp(-sigma * np.sqrt(delta_T) + (r - 0.5 * np.square(sigma)) * delta_T)
    p_tilda = (np.exp(r * delta_T) - d) / (u - d)
    q_tilda = 1 - p_tilda

    # To check whether no-arbitrage condition is satisfied we need to check whether: 0 < d < e^r < u holds. (here r is
    # the interest rate for the duration of one time step)
    condition1 = 0 < d
    condition2 = d < np.exp(r * delta_T)
    condition3 = np.exp(r * delta_T) < u

    if condition1 and condition2 and condition3:
        cnt_5 += 1 # Keep track of all configs for which no-arbitrage is satisfied.
    else:
        print(f"For M = {M}, \tno-arbitrage condition not satisfied.")
        break   # Or stop computing for that config.

    C_list, P_list = list(), list()

    for i in range(M + 1):
        S_price = S_0 * (u ** (M - i)) * (d ** i)
        C_price = max(0, S_price - K)
        P_price = max(0, K - S_price)
        C_list.append(C_price)
        P_list.append(P_price)

    for i in range(M):
        C_list_new, P_list_new = list(), list()
        for j in range(len(C_list) - 1):
            C_new = np.exp(-r * delta_T) * (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
            C_list_new.append(C_new)

            P_new = np.exp(-r * delta_T) * (P_list[j] * p_tilda + P_list[j + 1] * q_tilda)
            P_list_new.append(P_new)
        P_list, C_list = P_list_new, C_list_new

    C_prices_5.append(C_list[0])
    P_prices_5.append(P_list[0])

prices_5 = {"M": M_list_5,
            "Call": C_prices_5,
            "Put": P_prices_5}

if cnt_5 == len(M_list_5):
    print("No-arbitrage condition satisfied for all configurations.\n")
else:
    print("The above mentioned configurations failed no-arbitrage condition.\n")

print(f"Call_option price:  {prices_5['Call'][-1]}")
print(f"Put option price: {prices_5['Put'][-1]}")

plt.plot(prices_1["M"], prices_1["Call"])
plt.xlabel('M')
plt.ylabel('Initial Call Prices')
plt.title("Call with step size 1")
plt.show()

plt.plot(prices_1["M"], prices_1["Put"])
plt.xlabel('M')
plt.ylabel('Initial Put Prices')
plt.title("Put with step size 1")
plt.show()

plt.plot(prices_5["M"], prices_5["Call"])
plt.xlabel('M')
plt.ylabel('Initial Call Prices')
plt.title("Call with step size 5")
plt.show()

plt.plot(prices_5["M"], prices_5["Put"])
plt.xlabel('M')
plt.ylabel('Initial Put Prices')
plt.title("Put with step size 5")
plt.show()
