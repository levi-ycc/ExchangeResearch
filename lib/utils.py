import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import datetime
from lib.performance import basic, max_return_drawdown, sharpe, sortino


class FeatureEngineer:
    def __init__(self, windows=[5,10,15]):
        self.windows = windows
        self.start = 4
        
    def add_ta(self):
        self.add_rsi(self.windows)
        self.add_ema(self.windows)
        self.add_atr(self.windows)
        self.add_return()
        self.transformed.dropna(0, inplace=True)
        self.transformed['rsi5/rsi10']=self.transformed['rsi5']/self.transformed['rsi10']
        self.transformed['rsi10/rsi15']=self.transformed['rsi10']/self.transformed['rsi15']
        self.transformed['ema5/ema10']=self.transformed['ema5']/self.transformed['ema10']
        self.transformed['ema10/ema15']=self.transformed['ema10']/self.transformed['ema15']
        self.transformed['atr5/atr10']=self.transformed['atr5']/self.transformed['atr10']
        self.transformed['atr10/atr15']=self.transformed['atr10']/self.transformed['atr15']
        self.columns = self.transformed.columns
#         self.transformed.index = range(self.transformed.shape[0])
        
    def add_return(self):
#         self.transformed['pnl_return'] = self.transformed['C'].diff()/self.transformed['C'].shift(1)
#         self.transformed['pnl_return'][-1] = 0
        self.transformed['price_ratio'] = self.transformed['C']/self.transformed['C'].shift(1)
        self.transformed['open_ratio'] = self.transformed['O']/self.transformed['O'].shift(1)
        self.transformed['low_ratio'] = self.transformed['L']/self.transformed['L'].shift(1)
        self.transformed['high_ratio'] = self.transformed['H']/self.transformed['H'].shift(1)
#         self.transformed['price_ratio'][-1] = 0

        
    def add_rsi(self, period):
        for p in period:
            ta = talib.RSI(self.transformed['C'], p)
            self.transformed['rsi'+str(p)] = ta
            self.start+=1
        
    def add_ema(self, period):
        for p in period:
            ta = talib.EMA(self.transformed['C'], p)
            self.transformed['ema'+str(p)] = ta
            self.start+=1
            
    def add_atr(self, period):
        for p in period:
            ta = talib.ATR(self.transformed['H'], self.transformed['L'], self.transformed['C'], p)
            self.transformed['atr'+str(p)]=ta
            self.start+=1
        
    def fit(self, df):
        self.df = df.copy()
        self.transformed = self.df[['O','H','L','C','V']].copy()
        self.add_ta()
        self.transformed = self.transformed[self.columns[self.start:]]
        self.mean = np.mean(self.transformed.values, 0)
        self.std = np.std(self.transformed.values, 0)
        
    def transform(self, df=None, change_scale=0):
        mean = self.mean
        std = self.std
        
        if df is not None:
            self.start = 5
            self.df = df.copy()
            self.transformed = self.df[['O','H','L','C','V']].copy()
            self.add_ta()
            self.transformed = self.transformed[self.columns[self.start:]]
            
            if change_scale:
                mean = np.mean(self.transformed.values[:change_scale,:], 0)
                std = np.std(self.transformed.values[:change_scale,:], 0)
                self.transformed.drop([i+self.transformed.index[0] for i in range(change_scale)] , inplace=True)
        
        tmp_df = self.transformed.copy()
        tmp_df = (tmp_df-mean)/std
        return tmp_df
    
class BackTest:
    def __init__(self, data, model, env, change_scale=0, market='C', slippage=0):
        self.market = market
        self.slippage = slippage
        self.data = env.fe.transform(data, change_scale)
        self.trading_cost = env.trading_cost
        self.time_cost = env.time_cost
        self.date = data.loc[env.shift+change_scale:, 'Date'].copy()
        self.time = data.loc[env.shift+change_scale:].get('Time', self.date).copy()
        self.date.index = range(self.date.shape[0])
        self.time.index = range(self.time.shape[0])
        self.raw = data[-self.data.shape[0]:]
        self.raw.index = range(self.data.shape[0])
        self.data.index = range(self.data.shape[0])
        self.convert_to_array()
        self.model = model
        self.row_num = self.array.shape[0]
        self.discrete_action_dict = {'0':'1', '1':'-1', '2':'0'}
        self.predict()
        self.calculate_profit()
        
    def convert_to_array(self):
        self.array = self.data.values
    
    def predict(self):
        pos_list = []
        for idx in range(self.row_num):
            pos_list.append(np.argmax(self.model.compute_q_values(self.array[[idx]])))
        pos_list = [int(self.discrete_action_dict.get(str(p))) for p in pos_list]
        self.data['Position'] = pos_list
        
        
    def __out(self):
        self.data['Date'] = self.date
        self.data['Time'] = self.time
        self.data[['O','H','L','C','V']] = self.raw[['O','H','L','C','V']]
        return self.data
    
    def out(self):
        return self.__out()
    
    def calculate_profit(self):
        last_pos = 0
        if self.market == 'C':
            market = self.raw['C']
        elif self.market == 'O':
            market = self.raw['O'].shift(-1).ffill()
        else:
            market = mid_price(self.raw)
        
        market_nxt = market.shift(-1).ffill().copy()
        market_nxt.index = market.index
        market.loc[market!=market_nxt] += self.slippage
        
        pnl = self.data['Position']*market.diff().shift(-1).fillna(0)
        self.data['PnL'] = pnl
        self.data['Cum_PnL'] = self.data['PnL'].cumsum()
        self.data['Return'] = self.data['PnL']/market
        
    def compute_pnl(self, pos, delta_pos, price, price_n):
        diff = mid_price(price_n) - mid_price(price)
        return pos*diff-delta_pos*self.trading_cost*mid_price(price)-self.time_cost
        
    def save_csv(self, fn):
        saved_data = self.data.copy()
        saved_data[['O','H','L','C','V']] = self.raw[['O','H','L','C','V']]
        saved_data.to_csv('journal/{}'.format(fn))
        
    def stats(self):
        basic(self.data['Return'])
        max_return_drawdown(self.data['Return'])
        sharpe(self.data['Return'])
        sortino(self.data['Return'])
        
    
def mid_price(df):
    return (df['H']+df['L'])/2

def plot(journal_df, start_index=0, save_img=True):
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    last_pos=0
    for i in range(start_index,journal_df.shape[0],1):
        pos = journal_df['Position'].iloc[i]
        if last_pos != pos:
            if pos < 0 :
                ax.scatter(x=journal_df.index[i], y=journal_df['C'].iloc[i], color='red')
            elif pos > 0:
                ax.scatter(x=journal_df.index[i], y=journal_df['C'].iloc[i], color='green')
            else:
                ax.scatter(x=journal_df.index[i], y=journal_df['C'].iloc[i], color='yellow')
            last_pos = pos
    ax.plot(journal_df['C'][start_index:])
    
    if save_img:
        fig.savefig('img/'+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'.png')
