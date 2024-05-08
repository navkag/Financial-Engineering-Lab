import matplotlib.pyplot as plt
import numpy as np


def vasicek_model(term, n_units, mode=0):
    beta, mu, sigma, r = term
    if not mode:
        times = np.linspace(1e-5, 10, n_units)
    else:
        times = np.linspace(0.1, 500, n_units)

    yields = list()
    t = 0

    for T in times:
        B = (1 - np.exp(-beta * (T - t))) / beta
        A = (B - (T - t)) * (beta * beta * mu -
                             0.5 * (sigma ** 2)) / (beta ** 2)
        A -= (sigma ** 2) * (B ** 2) / (4 * beta)
        P = np.exp(A - B * r)
        y = - np.log(P) / (T - t)
        yields.append(y)

    return times, yields


def main():
    terms = [[5.9, 0.2, 0.3, 0.1], [3.9, 0.1, 0.3, 0.2], [0.1, 0.4, 0.11, 0.1]]

    for term in terms:
        T, yields = vasicek_model(term, 10)

        plt.plot(T, yields, marker='o')
        plt.xlabel('Maturity (T)')
        plt.ylabel('Yield')
        plt.title(f'Term structure for parameter set: {term}')
        # plt.show()
        plt.savefig(f'Term structure for parameter set: {term}.png')
        plt.clf()

        # Now for ten different sets of r_init for 500 time units.
        for i in range(1, 11):
            term[-1] = 0.1 * i
            T, yields = vasicek_model(term, 500)

            plt.plot(T, yields, label=f"r(0): {term[-1]:.2f}")
        plt.xlabel('Maturity (T)')
        plt.ylabel('Yield')
        plt.title(
            f'Term structure (500 units, 10 r(0)s) for parameter set: {term[:3]}')
        # plt.show()
        plt.legend()
        plt.savefig(
            f'Term structure (500 units, 10 r(0)s) for parameter set: {term[:3]}.png')
        plt.clf()


if __name__ == "__main__":
    main()
