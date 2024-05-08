import numpy as np
import pandas as pd

S_0, K, T, r, sigma, M = 100, 105, 5, 0.05, 0.4, 20
times = np.array([0, 0.50, 1, 1.50, 3, 4.5])
C_prices, P_prices = dict(), dict()


delta_T = T / M
notable_itr = times / delta_T
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
    print(f"No-arbitrage condition satisfied for M = {M}\n")
else:
    print(f"No-arbitrage condition not satisfied for M = {M}\n")
    exit() # Stop further calculations if no-arbitrage is violated.

C_list, P_list = list(), list()

for i in range(M + 1):
    S_price = S_0 * (u ** (M - i)) * (d ** i)
    C_price = max(0, S_price - K)
    P_price = max(0, K - S_price)
    C_list.append(C_price)
    P_list.append(P_price)

for i in range(M-1, -1, -1):
    C_list_new, P_list_new = list(), list()
    for j in range(len(C_list) - 1):
        C_new = np.exp(-r * delta_T) * (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
        C_list_new.append(C_new)

        P_new = np.exp(-r * delta_T) * (P_list[j] * p_tilda + P_list[j + 1] * q_tilda)
        P_list_new.append(P_new)
    P_list, C_list = P_list_new, C_list_new
    if i in notable_itr:
        C_prices[str(i)], P_prices[str(i)] = C_list, P_list

for index, key in enumerate(C_prices):
    print(f"T = {int(key) * delta_T}")
    disp_df = pd.DataFrame({"Call": C_prices[key], "Put": P_prices[key]})
    print(disp_df)
    print()
