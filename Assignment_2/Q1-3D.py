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
                u = np.exp(sigma  * np.sqrt(delta_T))
                d = np.exp(-sigma  * np.sqrt(delta_T))
        else:
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
                        C_new = np.exp(-r * delta_T) * (C_list[j] * p_tilda + C_list[j + 1] * q_tilda)
                        C_list_new.append(C_new)

                        P_new = np.exp(-r * delta_T) * (P_list[j] * p_tilda + P_list[j + 1] * q_tilda)
                        P_list_new.append(P_new)
                P_list, C_list = P_list_new, C_list_new


        return [C_list[0], P_list[0]]


def main():
        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying S_0 prices.

        S_0_prices_1, S_0_prices_2  = list(), list()
        for S_0 in np.linspace(25, 200, 50):
                S_0_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
                S_0_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        S_0_prices_1 = np.array(S_0_prices_1)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_1[:, 0])
        plt.title("European Call Option Prices vs spot for part (a)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_1[:, 1])
        plt.title("European Put Option Prices vs spot for part (a)")
        plt.savefig('European Option Prices vs spot part (a)')
        plt.show()


        S_0_prices_2 = np.array(S_0_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_2[:, 0])
        plt.title("European Call Option Prices vs spot for part (b)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), S_0_prices_2[:, 1])
        plt.title("European Put Option Prices vs spot for part (b)")
        plt.savefig('European Option Prices vs spot part (b)')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # Varying K prices.

        K_prices_1, K_prices_2  = list(), list()
        for K in np.linspace(25, 200, 50):
                K_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
                K_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        K_prices_1 = np.array(K_prices_1)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), K_prices_1[:, 0])
        plt.title("European Call Option Prices vs strike for part (a)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), K_prices_1[:, 1])
        plt.title("European Put Option Prices vs strike for part (a)")
        plt.savefig('European Option Prices vs strike part (a)')
        plt.show()


        K_prices_2 = np.array(K_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(25, 200, 50), K_prices_2[:, 0])
        plt.title("European Call Option Prices vs strike for part (b)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(25, 200, 50), K_prices_2[:, 1])
        plt.title("European Put Option Prices vs strike for part (b)")
        plt.savefig('European Option Prices vs strike part (b)')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying r values.

        r_prices_1, r_prices_2  = list(), list()
        for r in np.linspace(1e-4, 1, 50):
                r_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
                r_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        r_prices_1 = np.array(r_prices_1)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_1[:, 0])
        plt.title("European Call Option Prices vs rate for part (a)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_1[:, 1])
        plt.title("European Put Option Prices vs rate for part (a)")
        plt.savefig('European Option Prices vs rate for part (a)')
        plt.show()

        r_prices_2 = np.array(r_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_2[:, 0])
        plt.title("European Call Option Prices vs rate for part (b)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-4, 1, 50), r_prices_2[:, 1])
        plt.title("European Put Option Prices vs rate for part (b)")
        plt.savefig('European Option Prices vs rate for part (b)')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying sigma values.

        sigma_prices_1, sigma_prices_2  = list(), list()
        for sigma in np.linspace(1e-2, 1, 50):
                sigma_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
                sigma_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

        sigma_prices_1 = np.array(sigma_prices_1)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_1[:, 0])
        plt.title("European Call Option Prices vs volatility for part (a)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_1[:, 1])
        plt.title("European Put Option Prices vs volatility for part (a)")
        plt.savefig('European Option Prices vs volatility for part (a)')
        plt.show()

        sigma_prices_2 = np.array(sigma_prices_2)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_2[:, 0])
        plt.title("European Call Option Prices vs volatility for part (b)")

        plt.subplot(1, 2, 2)
        plt.plot(np.linspace(1e-2, 1, 50), sigma_prices_2[:, 1])
        plt.title("European Put Option Prices vs volatility for part (b)")
        plt.savefig('European Option Prices vs volatility for part (b)')
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100
        # Varying M values.
        K_list = [95, 100, 105]

        for K in K_list:
          M_prices_1, M_prices_2 = list(),  list()
          for M in range(50, 201):
            M_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
            M_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))

          M_prices_1 = np.array(M_prices_1)
          M_prices_2 = np.array(M_prices_2)

          plt.figure(figsize=(15, 4))
          plt.subplot(1, 2, 1)
          plt.plot(np.arange(50, 201), M_prices_1[:, 0])
          plt.title(f"European Call Option Prices vs #Sub-intervals for part (a) with K: {K}")

          plt.subplot(1, 2, 2)
          plt.plot(np.arange(50, 201), M_prices_1[:, 1])
          plt.title(f"European Put Option Prices vs #Sub-intervals for part (a) with K: {K}")
          plt.savefig(f"European Option Prices versus #Sub-intervals for part (a) with K: {K}")
          plt.show()


          plt.figure(figsize=(15, 4))
          plt.subplot(1, 2, 1)
          plt.plot(np.arange(50, 201), M_prices_2[:, 0])
          plt.title(f"European Call Option Prices vs #Sub-intervals for part (b) with K: {K}")

          plt.subplot(1, 2, 2)
          plt.plot(np.arange(50, 201), M_prices_2[:, 1])
          plt.title(f"European Put Option Prices vs #Sub-intervals for part (b) with K: {K}")
          plt.savefig(f"European Option Prices versus #Sub-intervals for part (b) with K: {K}")
          plt.show()

        # Now plotting 3D plots.

        # plot S_0 K plots.
        S_0 =  np.linspace(25, 200, 50)
        K =  np.linspace(25, 200, 50)
        S_0_K_prices_1, S_0_K_prices_2  = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
          for k in K:
            S_0_K_prices_1.append(generate_prices(s, k, r, sigma, M, T, 1))
            S_0_K_prices_2.append(generate_prices(s, k, r, sigma, M, T, 2))
            x_axis.append(s)
            y_axis.append(k)

        S_0_K_prices_1, S_0_K_prices_2 = np.array(S_0_K_prices_1), np.array(S_0_K_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_K_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and K for part (a)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("K")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_K_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and K for part (a)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("K")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and K for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_K_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and K for part (b)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("K")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_K_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and K for part (b)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("K")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and K for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot S_0 r plots.
        S_0 = np.linspace(25, 200, 50)
        R = np.linspace(1e-4, 1, 50)
        S_0_r_prices_1, S_0_r_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
                for r in R:
                        S_0_r_prices_1.append(generate_prices(s, K, r, sigma, M, T, 1))
                        S_0_r_prices_2.append(generate_prices(s, K, r, sigma, M, T, 2))
                        x_axis.append(s)
                        y_axis.append(r)

        S_0_r_prices_1, S_0_r_prices_2 = np.array(S_0_r_prices_1), np.array(S_0_r_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_r_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and rate for part (a)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("r")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_r_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and rate for part (a)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("r")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and rate for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_r_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and rate for part (b)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("r")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_r_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and rate for part (b)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("r")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and rate for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot S_0 sigma plots.
        S_0 = np.linspace(25, 200, 50)
        Sigma = np.linspace(1e-2, 1, 50)
        S_0_sigma_prices_1, S_0_sigma_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
                for sigma in Sigma:
                        S_0_sigma_prices_1.append(generate_prices(s, K, r, sigma, M, T, 1))
                        S_0_sigma_prices_2.append(generate_prices(s, K, r, sigma, M, T, 2))
                        x_axis.append(s)
                        y_axis.append(sigma)

        S_0_sigma_prices_1, S_0_sigma_prices_2 = np.array(S_0_sigma_prices_1), np.array(S_0_sigma_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_sigma_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and volatility for part (a)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_sigma_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and volatility for part (a)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and volatility for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_sigma_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs S_0 and volatility for part (b)")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_sigma_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs S_0 and volatility for part (b)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs S_0 and volatility for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 95, 1, 0.08, 0.3, 100

        # plot S_0 M plots with K = 95
        S_0 = np.linspace(25, 200, 50)
        M = [i for i in range(50, 101)]
        S_0_M_prices_1, S_0_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
                for m in M:
                        S_0_M_prices_1.append(generate_prices(s, K, r, sigma, m, T, 1))
                        S_0_M_prices_2.append(generate_prices(s, K, r, sigma, m, T, 2))
                        x_axis.append(s)
                        y_axis.append(m)

        S_0_M_prices_1, S_0_M_prices_2 = np.array(S_0_M_prices_1), np.array(S_0_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot S_0 M plots with K = 100
        S_0 = np.linspace(25, 200, 50)
        M = [i for i in range(50, 101)]
        S_0_M_prices_1, S_0_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
                for m in M:
                        S_0_M_prices_1.append(generate_prices(s, K, r, sigma, m, T, 1))
                        S_0_M_prices_2.append(generate_prices(s, K, r, sigma, m, T, 2))
                        x_axis.append(s)
                        y_axis.append(m)

        S_0_M_prices_1, S_0_M_prices_2 = np.array(S_0_M_prices_1), np.array(S_0_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 105, 1, 0.08, 0.3, 100

        # plot S_0 M plots with K = 105
        S_0 = np.linspace(25, 200, 50)
        M = [i for i in range(50, 101)]
        S_0_M_prices_1, S_0_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for s in S_0:
                for m in M:
                        S_0_M_prices_1.append(generate_prices(s, K, r, sigma, m, T, 1))
                        S_0_M_prices_2.append(generate_prices(s, K, r, sigma, m, T, 2))
                        x_axis.append(s)
                        y_axis.append(m)

        S_0_M_prices_1, S_0_M_prices_2 = np.array(S_0_M_prices_1), np.array(S_0_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("S_0")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, S_0_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs S_0 and #sub-intervals for part (b) with K: {K}")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot K r plots.
        K = np.linspace(25, 200, 50)
        R = np.linspace(1e-4, 1, 50)
        K_r_prices_1, K_r_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for k in K:
                for r in R:
                        K_r_prices_1.append(generate_prices(S_0, k, r, sigma, M, T, 1))
                        K_r_prices_2.append(generate_prices(S_0, k, r, sigma, M, T, 2))
                        x_axis.append(k)
                        y_axis.append(r)

        K_r_prices_1, K_r_prices_2 = np.array(K_r_prices_1), np.array(K_r_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_r_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs K and rate for part (a)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("r")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_r_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs K and rate for part (a)")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("r")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs K and rate for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_r_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs K and rate for part (b)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("r")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_r_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs K and rate for part (b)")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("r")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs K and rate for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot K sigma plots.
        K = np.linspace(25, 200, 50)
        Sigma = np.linspace(1e-2, 1, 50)
        K_sigma_prices_1, K_sigma_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for k in K:
                for sigma in Sigma:
                        K_sigma_prices_1.append(generate_prices(S_0, k, r, sigma, M, T, 1))
                        K_sigma_prices_2.append(generate_prices(S_0, k, r, sigma, M, T, 2))
                        x_axis.append(k)
                        y_axis.append(sigma)

        K_sigma_prices_1, K_sigma_prices_2 = np.array(K_sigma_prices_1), np.array(K_sigma_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_sigma_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs K and volatility for part (a)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_sigma_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs K and volatility for part (a)")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs K and volatility for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_sigma_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs K and volatility for part (b)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_sigma_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs K and volatility for part (b)")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs K and volatility for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot K M plots
        K = np.linspace(25, 200, 50)
        M = [i for i in range(50, 101)]
        K_M_prices_1, K_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for k in K:
                for m in M:
                        K_M_prices_1.append(generate_prices(S_0, k, r, sigma, m, T, 1))
                        K_M_prices_2.append(generate_prices(S_0, k, r, sigma, m, T, 2))
                        x_axis.append(k)
                        y_axis.append(m)

        K_M_prices_1, K_M_prices_2 = np.array(K_M_prices_1), np.array(K_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs K and #sub-intervals for part (a)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs K and #sub-intervals for part (a)")
        axes[1].set_xlabel("S_0")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs K and #sub-intervals for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, K_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs K and #sub-intervals for part (b)")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, K_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs K and #sub-intervals for part (b)")
        axes[1].set_xlabel("K")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs K and #sub-intervals for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot r sigma plots.
        R = np.linspace(0.1, 0.9, 50)
        Sigma = np.linspace(0.1, 0.9, 50)
        R_sigma_prices_1, R_sigma_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for r in R:
                for sigma in Sigma:
                        R_sigma_prices_1.append(generate_prices(S_0, K, r, sigma, M, T, 1))
                        R_sigma_prices_2.append(generate_prices(S_0, K, r, sigma, M, T, 2))
                        x_axis.append(r)
                        y_axis.append(sigma)

        R_sigma_prices_1, R_sigma_prices_2 = np.array(R_sigma_prices_1), np.array(R_sigma_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_sigma_prices_1[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs rate and volatility for part (a)")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_sigma_prices_1[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs rate and volatility for part (a)")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs rate and volatility for part (a)")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_sigma_prices_2[:, 0], cmap='Greens')
        axes[0].set_title("European Call Option Prices vs rate and volatility for part (b)")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("volatility")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_sigma_prices_2[:, 1], cmap='Greens')
        axes[1].set_title("European Put Option Prices vs rate and volatility for part (b)")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("volatility")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig("European Option Prices vs rate and volatility for part (b)")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 95, 1, 0.08, 0.3, 100

        # plot rate M plots with K = 95
        R = np.linspace(0.1, 0.9, 50)
        M = [i for i in range(50, 101)]
        R_M_prices_1, R_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for r in R:
                for m in M:
                        R_M_prices_1.append(generate_prices(S_0, K, r, sigma, m, T, 1))
                        R_M_prices_2.append(generate_prices(S_0, K, r, sigma, m, T, 2))
                        x_axis.append(r)
                        y_axis.append(m)

        R_M_prices_1, R_M_prices_2 = np.array(R_M_prices_1), np.array(R_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 100, 1, 0.08, 0.3, 100

        # plot rate M plots with K = 100
        R = np.linspace(0.1, 0.9, 50)
        M = [i for i in range(50, 101)]
        R_M_prices_1, R_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for r in R:
                for m in M:
                        R_M_prices_1.append(generate_prices(S_0, K, r, sigma, m, T, 1))
                        R_M_prices_2.append(generate_prices(S_0, K, r, sigma, m, T, 2))
                        x_axis.append(r)
                        y_axis.append(m)

        R_M_prices_1, R_M_prices_2 = np.array(R_M_prices_1), np.array(R_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        plt.show()

        S_0, K, T, r, sigma, M = 100, 105, 1, 0.08, 0.3, 100

        # plot rate M plots with K = 105
        R = np.linspace(0.1, 0.9, 50)
        M = [i for i in range(50, 101)]
        R_M_prices_1, R_M_prices_2 = list(), list()
        x_axis, y_axis = list(), list()

        for r in R:
                for m in M:
                        R_M_prices_1.append(generate_prices(S_0, K, r, sigma, m, T, 1))
                        R_M_prices_2.append(generate_prices(S_0, K, r, sigma, m, T, 2))
                        x_axis.append(r)
                        y_axis.append(m)

        R_M_prices_1, R_M_prices_2 = np.array(R_M_prices_1), np.array(R_M_prices_2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_1[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_1[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (a) with K: {K}")
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

        axes[0].scatter3D(x_axis, y_axis, R_M_prices_2[:, 0], cmap='Greens')
        axes[0].set_title(f"European Call Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[0].set_xlabel("rate")
        axes[0].set_ylabel("M")
        axes[0].set_zlabel("C_0")

        axes[1].scatter3D(x_axis, y_axis, R_M_prices_2[:, 1], cmap='Greens')
        axes[1].set_title(f"European Put Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        axes[1].set_xlabel("rate")
        axes[1].set_ylabel("M")
        axes[1].set_zlabel("P_0")

        plt.tight_layout()

        plt.savefig(f"European Option Prices vs rate and #sub-intervals for part (b) with K: {K}")
        plt.show()

        # sigma M 3d plot

        for K in [95, 100, 105]:
                S_0, T, r, sigma, M = 100, 1, 0.08, 0.3, 100

                # plot rate M plots with K = 95
                Sigma = np.linspace(0.1, 0.9, 50)
                M = [i for i in range(50, 101)]
                sigma_M_prices_1, sigma_M_prices_2 = list(), list()
                x_axis, y_axis = list(), list()

                for sigma in Sigma:
                        for m in M:
                                sigma_M_prices_1.append(generate_prices(S_0, K, r, sigma, m, T, 1))
                                sigma_M_prices_2.append(generate_prices(S_0, K, r, sigma, m, T, 2))
                                x_axis.append(sigma)
                                y_axis.append(m)

                sigma_M_prices_1, sigma_M_prices_2 = np.array(sigma_M_prices_1), np.array(sigma_M_prices_2)

                fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

                axes[0].scatter3D(x_axis, y_axis, sigma_M_prices_1[:, 0], cmap='Greens')
                axes[0].set_title(f"European Call Option Prices vs volatility and #sub-intervals for part (a) with K: {K}")
                axes[0].set_xlabel("volatility")
                axes[0].set_ylabel("M")
                axes[0].set_zlabel("C_0")

                axes[1].scatter3D(x_axis, y_axis, sigma_M_prices_1[:, 1], cmap='Greens')
                axes[1].set_title(f"European Put Option Prices vs volatility and #sub-intervals for part (a) with K: {K}")
                axes[1].set_xlabel("volatility")
                axes[1].set_ylabel("M")
                axes[1].set_zlabel("P_0")

                plt.tight_layout()

                plt.savefig(f"European Option Prices vs volatility and #sub-intervals for part (a) with K: {K}")
                plt.show()

                fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})

                axes[0].scatter3D(x_axis, y_axis, sigma_M_prices_2[:, 0], cmap='Greens')
                axes[0].set_title(f"European Call Option Prices vs volatility and #sub-intervals for part (b) with K: {K}")
                axes[0].set_xlabel("volatility")
                axes[0].set_ylabel("M")
                axes[0].set_zlabel("C_0")

                axes[1].scatter3D(x_axis, y_axis, sigma_M_prices_2[:, 1], cmap='Greens')
                axes[1].set_title(f"European Put Option Prices vs volatility and #sub-intervals for part (b) with K: {K}")
                axes[1].set_xlabel("volatility")
                axes[1].set_ylabel("M")
                axes[1].set_zlabel("P_0")

                plt.tight_layout()

                plt.savefig(f"European Option Prices vs volatility and #sub-intervals for part (b) with K: {K}")
                plt.show()

main()