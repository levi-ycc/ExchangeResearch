from strategy.abstract import StrategyAbstract
from util.performance import basic, max_return_drawdown, sharpe, sortino
from utils import mid_price
import pandas as pd

class Strategy(StrategyAbstract):
    
    def __init__(self, journal_df):
        if 'Date' in journal_df.columns:
            self.strat_df = journal_df[['Date', 'Time', 'O', 'H', 'L', 'C', 'V', 'Position']].copy()
        else:
            self.strat_df = journal_df[['O', 'H', 'L', 'C', 'V','Position']].copy()
        
    def parse_data(self, market='M'):
        if market == 'C':
            self.strat_df['market'] = self.strat_df['C']
        elif market == 'O':
            self.strat_df['market'] = self.strat_df['O'].shift(-1).ffill()
        else:
            self.strat_df['market'] = mid_price(self.strat_df)
        
        
    def run_backtest(self):
        self.strat_df['PnL'] = self.strat_df['Position'] * self.strat_df['market'].diff().shift(-1)
        self.strat_df['PnL'].fillna(0, inplace=True)
        self.strat_df['CumPnL'] = self.strat_df['PnL'].cumsum()
        self.strat_df['Return'] = self.strat_df['PnL']/self.strat_df['market']
        
    
    def stats(self):
        returns = self.strat_df['Return']
        basic(returns)
        max_return_drawdown(returns)
        sharpe(returns)
        sortino(returns)
    
    def out_journal(self, save_name=None):
        self.strat_df.index = range(self.strat_df.shape[0])
        if save_name is None:
            self.strat_df.to_csv('journal/strat_journal.csv', index=False)
        else:
            self.strat_df.to_csv('journal/{}.csv'.format(save_name), index=False)
            
class MyStrategy(Strategy):
    
    def __init__(self, journal_df, market='M'):
        super().__init__(journal_df)
        self.parse_data(market)
        
    def generate_signals(self, p50):
        pass
        
def stats_result(strat_list, training_stats):
    if type(strat_list) == list:
        assert len(strat_list) == len(training_stats)
        for d in range(len(strat_list)):
            ns = MyStrategy(strat_list[d])
            ns.generate_signals(training_stats[d])
            ns.run_backtest()
            ns.stats()
            print('CumSum: {}'.format(ns.strat_df['PnL'].sum()))
        return 0
    
    else:
        ns = MyStrategy(strat_list)
        ns.generate_signals(training_stats)
        ns.run_backtest()
        ns.stats()
        print('CumSum: {}'.format(ns.strat_df['PnL'].sum()))
        return ns.strat_df