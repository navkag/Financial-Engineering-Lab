import numpy as np
import matplotlib.pyplot as plt
import time


def optimizer(itr, u, d, M, p_tilda, q_tilda, discount_term, spot, K, price_history, idx):
        if itr == M + 1 or (spot, K) in price_history[itr]:
                return

        optimizer(itr + 1, u, d, M, p_tilda, q_tilda, discount_term, spot *
                  u, K, price_history, idx + "u")
        optimizer(itr + 1, u, d, M, p_tilda, q_tilda, discount_term, spot *
                  d, K, price_history, idx + "d")

        if itr == M:
                price_history[M][(spot, K)] = [
                    max(0, spot - K), idx]
        else:
                price_history[itr][(spot, K)] = [discount_term * (p_tilda * price_history[itr + 1][(
                    spot * u, K)][0] + q_tilda * price_history[itr + 1][(spot * d, K)][0]), idx]


def generate_prices(S_0, r, sigma, K, M, T, mode, display):
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
        else:
                print(f"For M = {M}, \tno-arbitrage condition not satisfied.")
                return

        price_history = list()
        for i in range(M + 1):
                price_history.append(dict())

        discount_term = np.exp(-r * delta_T)
        optimizer(0, u, d, M, p_tilda, q_tilda,
                  discount_term, S_0, K, price_history, "S0")

        return price_history[0][(S_0, S_0)] if not display else price_history


def generate_prices_binomial(S_0, r, sigma, K, M, T, mode, display):
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
        else:
                print(f"For M = {M}, \tno-arbitrage condition not satisfied.")
                return

        C_list, P_list = list(), list()

        # Get option prices over the life of the option.
        history = list()

        sums = list()

        for i in range(int(pow(2, M))):
                S = S_0
                path = format(i, 'b').zfill(M)

                for num in path:
                        S *= u if num == '0' else d
                sums.append(S)

        S_price = np.array(sums)

        for i in range(len(sums)):
                C_price = max(0, S_price[i] - K)
                C_list.append(C_price)

        history.append(C_list)

        for i in range(M, 0, -1):
                C_list_new = list()
                for j in range(0, int(pow(2, i)), 2):
                        C_new = np.exp(-r * delta_T) * \
                            (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
                        C_list_new.append(C_new)

                history.append(C_list_new)
                C_list = C_list_new

        return C_list[0] if not display else history


def main():
        option_type = "European"
        S_0, T, r, sigma, K = 100, 1, 0.08, 0.3, 100

        M_list = [5, 10, 25, 50]

        print(f"\nBinomial method.\n")

        for M in M_list[:3]:
                t1 = time.time()
                print(f"{option_type} Call Option Prices for M={M}: {generate_prices_binomial(S_0, r, sigma, K, M, T, 2, display=0)}")
                t2 = time.time()
                print(f"Time taken: {t2 - t1}")

        print(f"\nEfficient method.\n")

        for M in M_list:
                t1 = time.time()
                print(f"{option_type} Call Option Prices for M={M}: {generate_prices(S_0, r, sigma, K, M, T, 2, display=0)[0]}")
                t2 = time.time()
                print(f"Time taken: {t2 - t1}")

        print(f"\nIntermediate {option_type} option prices for M = 5\n")

        M = 5
        history = generate_prices(S_0, r, sigma, K, M, T, 2, display=1)

        for t in range(len(history)):
                print(f"At step {5 - t}")
                prices = list(history[5 - t].values())
                for i in range(len(prices)):
                        u_count, d_count = prices[i][1].count(
                            'u'), prices[i][1].count('d')
                        print(f"S_0.u^{u_count}.d^{d_count}: {prices[i][0]}")
                print()


main()
