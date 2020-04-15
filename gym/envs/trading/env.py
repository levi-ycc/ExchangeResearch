import gym
from gym import spaces
from gym.utils import seeding
import pandas as pd
import numpy as np
from util import FeatureEngineer
from util import BackTest, mid_price
from util.performance import sharpe, sortino
import pprint
import matplotlib.pyplot as plt
import datetime

class TradingEnv(gym.Env):
    def __init__(self, csv_list, trading_cost = 0, time_cost = 0, cycle = 10, market='C'):
        self.market = market
        self.cycle_count = 0
        self.cycle = cycle
        self.csv_list = csv_list
        self._init_df()
                
        self.journal_df = pd.DataFrame()
        self.rew_list = []
        self.profit_list = []
        
        self.trading_cost = trading_cost
        self.time_cost = time_cost
        self.obs = self.processed_array[self.obs_ptr, :]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(1, self.processed_array.shape[-1]),
                                            dtype=np.float64)
        self.action_space = spaces.Discrete(3)
        self.pos = 0
        self.pp = pprint.PrettyPrinter()
        
    def _init_df(self):
        self.csv = np.random.choice(self.csv_list)
        try:
            del self.df, self.fe, self.processed_df, self.processed_array
        except:
            pass
        self.df = pd.read_csv(self.csv)
        if 'Date' not in self.df.columns:
            self.df['Date'] = range(self.df.shape[0])
        self.fe = FeatureEngineer()
        self.fe.fit(self.df)
        self.processed_df = self.fe.transform()
        self.shift = self.processed_df.index[0]
        self.obs_ptr = 0
        self.trade_ptr = self.obs_ptr+1
        self.end_ptr = int((self.df.shape[0]-self.shift)*0.8)
        self.processed_array = np.asarray(self.processed_df.copy())
#         self.market_price_array = np.asarray(mid_price(self.df).copy())
        if self.market == 'C':
            self.market_price_array = np.asarray(self.df['C'].copy())
        elif self.market == 'O':
            self.market_price_array = np.asarray(self.df['O'].shift(-1).ffill().copy())
        else:
            self.market_price_array = np.asarray(mid_price(self.df).copy())
        
        
    def _get_obs(self):
        self.obs_ptr += 1
        self.trade_ptr = self.obs_ptr+1
        return self.processed_array[self.obs_ptr, :]
    
    def get_obs(self):
        next_obs = self._get_obs()
        return next_obs
    
    def compute_profit(self, action):
        cost = self.compute_cost(action)
        #Buy
        if action == 0:
            self.pos = 1
        #Sell
        elif action == 1:
            self.pos = -1
        #Open trade
        else:
            self.pos = 0
        
        profit = self.pos*(self.market_price_array[self.trade_ptr+self.shift]- \
                           self.market_price_array[self.obs_ptr+self.shift])
        net = profit - cost
        return profit, net
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_cost(self, action):
        price = self.market_price_array[self.obs_ptr+self.shift]
        if self.pos == 1 and action == 1:
            return 2*self.trading_cost*price + self.time_cost
        elif self.pos == -1 and action == 0:
            return 2*self.trading_cost*price + self.time_cost
        elif action == 2 and self.pos == 1:
            return 0
        elif action == 2 and self.pos == -1:
            return 0
        else:
            return self.time_cost
        
        
    def _step(self, action):
        self.profit, reward = self.compute_profit(action)
        self.profit_list.append(self.profit)
        self.rew_list.append(reward)

        info = self.get_info()
        self.journal_df = self.journal_df.append(info, ignore_index=True)

        self.obs = self.get_obs()
        done = self.if_done()
        
        return self.obs, reward, done, info

    def step(self, action):
        observation, reward, done, info = self._step(action)
        return observation, reward, done, info
    
    def if_done(self):
        done = self.end_ptr == self.trade_ptr
        return done
    
    def get_info(self):
        ptr = self.obs_ptr+self.shift
        return {'Date':self.df.iloc[ptr]['Date'],
                'O':self.df.iloc[ptr]['O'],
                'H':self.df.iloc[ptr]['H'],
                'L':self.df.iloc[ptr]['L'],
                'C':self.df.iloc[ptr]['C'],
                'V':self.df.iloc[ptr]['V'],
                'Position': self.pos,
                'PnL':self.profit,
                'Cum_PnL':sum(self.profit_list),
                'Return': self.profit/abs(mid_price(self.df.iloc[ptr]))
                }
    
    def _reset(self):
        self.obs_ptr = 0
        self.trade_ptr = self.obs_ptr+1
        self.profit_list = []
        self.pos = 0
        self.journal_df = pd.DataFrame()
        self.cycle_count += 1
        if self.cycle_count % self.cycle == 0 and len(self.csv_list) > 1:
            self._init_df()
        
        self.obs = self.processed_array[self.obs_ptr, :]
        
        return self.obs
        
    def reset(self):
        obs = self._reset()
        return obs

    def _render(self, mode, close):
        if self.if_done():
            cum_pnl = sum(self.profit_list)
            std = np.std(self.profit_list)
            ret = self.journal_df['Return']
            print('\n')
            self.pp.pprint("Data: "+self.csv)
            self.pp.pprint("CumPnL: "+str(cum_pnl))
            self.pp.pprint("Sharpe: "+str(sharpe(ret, verbose=False, annual_risk_free_rate=0)))
            self.pp.pprint("Sortino: "+str(sortino(ret, verbose=False, annual_risk_free_rate=0)))
            print('\n')
