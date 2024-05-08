import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_prices(S_0, K, r, sigma, M, T, mode):
        '''
        if mode =  1, use u and d defined in (a)
        else (b)
        '''
        delta_T = T / M
        if mode == 1:
                u = np.exp(sigma * np.sqrt(delta_T))
                d = np.exp(-sigma * np.sqrt(delta_T))
        else:
                u = np.exp(sigma * np.sqrt(delta_T) +
                           (r - 0.5 * np.square(sigma)) * delta_T)
                d = np.exp(-sigma * np.sqrt(delta_T) +
                           (r - 0.5 * np.square(sigma)) * delta_T)
        p_tilda = (np.exp(r * delta_T) - d) / (u - d)
        q_tilda = 1 - p_tilda

        # To check whether no-arbitrage condition is satisfied we need to check whether: 0 < d < e^r < u holds. (here r is
        # the interest rate for the duration of one time step)
        condition1 = 0 < d
        condition2 = d < np.exp(r * delta_T)
        condition3 = np.exp(r * delta_T) < u

        if condition1 and condition2 and condition3:
                pass
                # print(f"For M = {M}, \tno-arbitrage condition satisfied.")
        else:
                print(f"For M = {M}, \tno-arbitrage condition not satisfied.")
                # break # Or stop computing for that config.
                return

        C_list, P_list = list(), list()

        for i in range(M + 1):
                S_price = S_0 * (u ** (M - i)) * (d ** i)
                C_price = max(0, S_price - K)
                P_price = max(0, K - S_price)
                C_list.append(C_price)
                P_list.append(P_price)

        for i in range(M, 0, -1):
                C_list_new, P_list_new = list(), list()
                for j in range(len(C_list) - 1):
                        S_price = S_0 * (u ** (i - 1 - j)) * (d ** j)

                        C_new = np.exp(-r * delta_T) * \
                            (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
                        C_new = max(S_price - K,  C_new)

                        C_list_new.append(C_new)

                        P_new = np.exp(-r * delta_T) * \
                            (P_list[j] * p_tilda + P_list[j + 1] * q_tilda)
                        P_new = max(K - S_price,  P_new)

                        P_list_new.append(P_new)
                P_list, C_list = P_list_new, C_list_new

        return [C_list[0], P_list[0]]


def main():
        option_type = "American"

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        [call, put] = generate_prices(S_0, K, r, sigma, M, T, 2)
        print(f"{option_type} call option price: {call}")
        print(f"{option_type} put option price: {put}")
        # Varying S_0 prices.

        S_0_prices_2 = list()
        for S_0 in np.linspace(25, 200, 50):
                S_0_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        S_0_prices_2 = np.array(S_0_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_2[:, 0])
        plt.title(f"{option_type} Call Option Prices vs spot.")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_2[:, 1])
        plt.title(f"{option_type} Put Option Prices vs spot.")
        plt.savefig(f'{option_type} Option Prices vs spot.')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # Varying K prices.

        K_prices_2 = list()
        for K in np.linspace(25, 200, 50):
                K_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        K_prices_2 = np.array(K_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), K_prices_2[:, 0])
        plt.title(f"{option_type} Call Option Prices vs strike.")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), K_prices_2[:, 1])
        plt.title(f"{option_type} Put Option Prices vs strike.")
        plt.savefig(f'{option_type} Option Prices vs strike.')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying r values.

        r_prices_2 = list()
        for r in np.linspace(1e-4, 1, 50):
                r_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        r_prices_2 = np.array(r_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_2[:, 0])
        plt.title(f"{option_type} Call Option Prices vs rate.")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_2[:, 1])
        plt.title(f"{option_type} Put Option Prices vs rate.")
        plt.savefig(f'{option_type} Option Prices vs rate.')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying sigma values.

        sigma_prices_1, sigma_prices_2 = list(), list()
        for sigma in np.linspace(1e-2, 1, 50):
                sigma_prices_2.append(generate_prices(
                    S_0, K, r, sigma, M, T, 2))

        sigma_prices_2 = np.array(sigma_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_2[:, 0])
        plt.title(f"{option_type} Call Option Prices vs volatility.")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_2[:, 1])
        plt.title(f"{option_type} Put Option Prices vs volatility.")
        plt.savefig(f'{option_type} Option Prices vs volatility.')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying M values.
        K_list = [95, 100, 105]

        for K in K_list:
                M_prices_1, M_prices_2 = list(),  list()
                for M in range(50, 201):
                        M_prices_2.append(generate_prices(
                            S_0, K, r, sigma, M, T, 2))

                M_prices_2 = np.array(M_prices_2)

                plt.figure(figsize=(15, 4))
                plt.subplot(1, 2, 1)
                plt.plot(np.arange(50, 201), M_prices_2[:, 0])
                plt.title(
                    f"{option_type} Call Option Prices vs #Sub-intervals with K: {K}")

                plt.subplot(1, 2, 2)
                plt.plot(np.arange(50, 201), M_prices_2[:, 1])
                plt.title(
                    f"{option_type} Put Option Prices vs #Sub-intervals with K: {K}")
                plt.savefig(
                    f"{option_type} Option Prices versus #Sub-intervals with K: {K}")
                plt.show()


main()
