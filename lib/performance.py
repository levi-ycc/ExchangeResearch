import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
Modified to matrix version of https://github.com/cnaimo/performance-analysis
"""

def basic(returns):
    results = returns.copy()  
    print('\n_______________________________________________________________________________________________________\n')
    if len(results) == 0:
        print('NO TRADES!')
        return

    results += 1
    win = results>1
    lose = ~win
    winners=np.sum(win)
    losers=returns.shape[0]-winners
    if losers:
        avg_loss = np.mean(results[lose])
    else:
        avg_loss = np.nan
        
    if winners:
        avg_win = np.mean(results[win])
    else:
        avg_win = np.nan

    print('\nAccuracy:\t' + str(round(100 * winners / (winners + losers), 2)) + '%',
          '\nTotal:\t\t' + str(results.shape[0]))
    print(
        'Avg Win:\t' + str(round(100 * (avg_win - 1), 2)) +
        '%\tMax Win:\t' + str(round(100 * (np.max(results[win]) - 1), 2)) + '%')
    if losers:
        print('Avg Loss:\t' + str(round(100 * (1 - avg_loss), 2)) + '%\tMax Loss:\t' +
              str(round(100 * (1 - np.min(results[lose])), 2)) + '%')
    else:
        print('No losses')


def max_return_drawdown(results, leverage=1, verbose=True):
    results = results.copy()
    results = np.asarray(results)
    results *= leverage
    results += 1
    max_dd = 1
    max_gain = 0

    gain = np.cumprod(results, dtype=float)
    max_gain_array = np.zeros(gain.shape[0])
    max_dd_array = np.zeros(gain.shape[0])
    
    for i in range(gain.shape[0]):
        if i == 0:
            max_gain_array[i] = gain[i]
        else:
            if max_gain_array[i-1] > gain[i]:
                max_gain_array[i] = max_gain_array[i-1]
            else:
                max_gain_array[i] = gain[i]
                        
    max_dd_array = gain/max_gain_array
    max_dd = np.min(max_dd_array)

    if verbose:
        print('Max Drawdown:', round(1 - max_dd, 4))
    return 1 - max_dd


def sharpe(results, leverage=1, annual_risk_free_rate=0.0, verbose=True):
    results = results.copy()
    daily_rfr = annual_risk_free_rate / 252
    returns = np.asarray(results)
    returns *= leverage
    sharpe = ((np.mean(returns) - daily_rfr) / np.std(returns)) * (252 ** 0.5)
    if verbose:
        print('Sharpe:', round(sharpe, 4))
    return sharpe


def sortino(results, leverage=1, annual_risk_free_rate=0.0, verbose=True):
    results = results.copy()
    daily_rfr = annual_risk_free_rate / 252
    returns = np.asarray(results)
    returns *= leverage
    downside = np.zeros(returns.shape[0])
    
    neg_index = (returns-daily_rfr) < 0
    downside[neg_index] = returns[neg_index]**2

    downside_risk = np.mean(downside) ** 0.5
    sortino = ((np.mean(returns) - daily_rfr) / downside_risk) * (252 ** 0.5)
    if verbose:
        print('Sortino:', round(sortino, 4))
    return sortino
