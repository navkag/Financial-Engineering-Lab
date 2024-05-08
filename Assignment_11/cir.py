import matplotlib.pyplot as plt
import numpy as np


def cir_model(term, n_units, mode=0):
    beta, mu, sigma, r = term
    if not mode:
        times = np.linspace(1e-5, 20, n_units)
    else:
        times = np.linspace(0.1, 600, n_units)

    yields = list()
    t = 0

    for T in times:
        x = T - t
        gamma = (beta ** 2 + 2 * (sigma ** 2)) ** 0.5
        term1 = np.exp(gamma * x) - 1
        term2 = term1 * (gamma + beta) + 2 * gamma

        B = 2 * term1 / term2

        A = 2 * gamma * np.exp((beta + gamma) * (x / 2))
        A /= term2
        A = A ** (2 * beta * mu / (sigma ** 2))

        P = A * np.exp(-B * r)

        y = - np.log(P) / x
        yields.append(y)

    return times, yields


def main():
    terms = [[0.02, 0.7, 0.02, 0.1], [
        0.7, 0.1, 0.3, 0.2], [0.06, 0.09, 0.5, 0.02]]

    for term in terms:
        T, yields = cir_model(term, 10)

        plt.plot(T, yields, marker='o')
        plt.xlabel('Maturity (T)')
        plt.ylabel('Yield')
        plt.title(f'Term structure for parameter set: {term}')
        # plt.show()
        plt.savefig(f'Term structure for parameter set: {term}.png')
        plt.clf()

        # Now for ten different sets of r_init for 600 time units.

        if terms.index(term) == 0:
            for i in range(1, 11):
                term[-1] = 0.1 * i
                T, yields = cir_model(term, 600, 1)

                plt.plot(T, yields, label=f"r(0): {term[-1]:.2f}")
            plt.xlabel('Maturity (T)')
            plt.ylabel('Yield')
            plt.title(
                f'Term structure (600 units, 10 r(0)s) for parameter set: {term[:3]}')
            # plt.show()
            plt.legend()
            plt.savefig(
                f'Term structure (600 units, 10 r(0)s) for parameter set: {term[:3]}.png')
            plt.clf()


if __name__ == "__main__":
    main()
