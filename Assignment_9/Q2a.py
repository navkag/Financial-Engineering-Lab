import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from datetime import datetime


def plot_3D(filename, opt_type):
  mpl.rcParams.update(mpl.rcParamsDefault)
  df = pd.read_csv(filename)

  # Option price vs strike vs maturity.
  # Each row has 25% chance of being added to this list.
  np.random.seed(45)
  mask = np.random.rand(len(df)) <= 0.25
  filtered_df = df[mask]

  strike = filtered_df["strike price"].values
  call = filtered_df["call price"].values
  put = filtered_df["put price"].values
  maturity = list()

  for i in range(len(filtered_df)):
    maturity.append((datetime.strptime(filtered_df["expiry"].values[i], '%d-%b-%y') -
                     datetime.strptime(filtered_df["date"].values[i], '%d-%b-%y')).days)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(strike, maturity, call, marker='.')
  ax.set_xlabel('Strike Price')
  ax.set_ylabel('Maturity (in days)')
  ax.set_zlabel('Call Price')
  ax.set_title(f'{opt_type}: 3D plot Call Option.')
  plt.savefig(f'2a/{opt_type}: 3D plot Call Option.')
  # plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(strike, maturity, put, marker='.')
  ax.set_xlabel('Strike Price')
  ax.set_ylabel('Maturity (in days')
  ax.set_zlabel('Put Price')
  ax.set_title(f'{opt_type}: 3D plot Put Option.')
  plt.savefig(f'2a/{opt_type}: 3D plot Put Option.')
  # plt.show()


def plot_2D(filename, opt_type):
  plt.rcParams["figure.figsize"] = (13, 4)
  df = pd.read_csv(filename)

  # Option price vs strike.
  # Each row has 25% chance of being added to this list.
  np.random.seed(42)
  mask = np.random.rand(len(df)) <= 0.25
  filtered_df = df[mask]

  print(filtered_df.columns)

  x = filtered_df["strike price"].values
  call = filtered_df["call price"].values
  put = filtered_df["put price"].values

  plt.subplot(1, 2, 1)
  plt.scatter(x, call, marker='.')
  plt.xlabel('Strike Prices')
  plt.ylabel('Call Prices')
  plt.title(f'{opt_type}: Call price vs Strike Price')

  plt.subplot(1, 2, 2)
  plt.scatter(x, put, marker='.')
  plt.xlabel('Strike price')
  plt.ylabel('Put Price')
  plt.title(f'{opt_type}: Put price vs Strike Price')
  plt.savefig(f'2a/{opt_type}: Option price vs Strike Price.png')
  # plt.show()

  # Option price vs Maturity
  # Each row has 5% chance of being added to this list.
  np.random.seed(40)
  mask = np.random.rand(len(df)) <= 0.05
  filtered_df = df[mask]

  x = list()

  for i in range(len(filtered_df)):

    x.append((datetime.strptime(filtered_df["expiry"].values[i], '%d-%b-%y') -
              datetime.strptime(filtered_df["date"].values[i], '%d-%b-%y')).days)

  call = filtered_df["call price"].values
  put = filtered_df["put price"].values

  plt.subplot(1, 2, 1)
  plt.scatter(x, call, marker='.')
  plt.xlabel('Maturity (in days)')
  plt.ylabel('Call Price')
  plt.title(f'{opt_type}: Call price vs Maturity')

  plt.subplot(1, 2, 2)
  plt.scatter(x, put, marker='.')
  plt.xlabel('Maturity (in days)')
  plt.ylabel('Put Price')
  plt.title(f'{opt_type}: Put price vs Maturity')
  plt.savefig(f'2a/{opt_type}: Option price vs Maturity.png')
  # plt.show()


def main():
  files = ['NIFTYoptiondata', 'adanistockoptions', 'airtelstockoptions', 'bajajstockoptions',
           'bpclstockoptions', 'ciplastockoptions', 'hclstockoptions', 'axisbankstockoptions']
  opt_type = ['NSE Index', 'ADANIENT.NS', 'BHARTIARTL.NS',
              'BAJAJ-AUTO.NS', 'BPCL.NS', 'CIPLA.NS', 'HCLTECH.NS', 'AXISBANK.NS']

  # files = ['NIFTYoptiondata']
  # opt_type = ['NSE Index']

  for idx in range(len(files)):
    files[idx] = './Data/' + files[idx] + '.csv'
    plot_2D(files[idx], opt_type[idx])
    plot_3D(files[idx], opt_type[idx])


if __name__ == "__main__":
  main()
