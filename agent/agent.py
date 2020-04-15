import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
from rl.random import OrnsteinUhlenbeckProcess
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model

from agent.abstract import AgentAbstract
from wrapper import MyWrapper
import lib.utils as utils
import lib.model as Model
from lib.performance import basic, max_return_drawdown, sharpe, sortino
from lib.utils import BackTest
from lib.config import Config

import random
import tensorflow as tf

class Agent(AgentAbstract):
    
    def __init__(self, config, save_name=None):
        
        self.config = config
        self.save_name = save_name
        self.env_name = self.config.env.get('env_name', 'Trading-v0')
        self.csv_list = self.config.env.get('csv_list', ['data/txf1.csv'])
        self.seed = int(self.config.env.get('seed', 123))
        self.split_ratio = float(self.config.env.get('split_ratio', 0.8))
        self.trading_cost = float(self.config.env.get('trading_cost', 0.0))
        self.time_cost = float(self.config.env.get('time_cost', 0.0))
        self.market = self.config.env.get('market', 'C')
        self.slippage = self.config.env.get('slippage', 0)

        self.episodes = int(self.config.model.get('episodes', 1))
        self.eps = float(self.config.model.get('eps', 1e-2))
        self.target_model_update = float(self.config.model.get('target_model_update', 1e-3))
        self.lr = float(self.config.model.get('learning_rate', 1e-3))
        self.verbose = int(self.config.model.get('verbose', 2))
        self.memory_len = int(self.config.model.get('memory_len', 200))
        
        self.plot = self.config.agent.get('plot', False)
        self.start_index = int(self.config.agent.get('start_index', 1100))
        
        self.init_env()
        self.create_agent()

    def fit(self):
        self.agent.fit(self.env, nb_steps=self.steps_per_episode*self.episodes, visualize=True,
                       verbose=self.verbose, nb_max_episode_steps=None)
        self.save_weights(self.save_name)
        
    def test(self):
        self.agent.test(self.env, nb_episodes=1, visualize=True, nb_max_episode_steps=None)
        
    def save_model(self, save_name=None):
        if not save_name:
            self.model.save('models/dqn_{}_weights.h5f'.format(self.env_name), overwrite=True)
        else:
            self.model.save('models/{}.h5f'.format(save_name), overwrite=True)
        
    def plot(self, start_index=None):
        if not start_index:
            utils.plot(self.env.journal_df, start_index=self.start_index)
        else:
            utils.plot(self.env.journal_df, start_index=start_index)
        
    def stats(self, df_return_column=None):
        if df_return_column is None:
            basic(self.env.journal_df['Return'])
            max_return_drawdown(self.env.journal_df['Return'])
            sharpe(self.env.journal_df['Return'])
            sortino(self.env.journal_df['Return'])
            
        else:
            basic(df_return_column)
            max_return_drawdown(df_return_column)
            sharpe(df_return_column)
            sortino(df_return_column)
            
    def backtest(self, custom_dataset=None, change_scale=0):
        if not custom_dataset:
            self.bt_df = self.env.df.iloc[self.env.end_ptr-self.env.shift-change_scale:].copy()
            self.bt_df = BackTest(self.bt_df, self.agent, self.env, change_scale,
                                  self.market, self.slippage).out().copy()

        else:
            df = pd.read_csv(custom_dataset, index_col=False)
            self.bt_df = BackTest(df, self.agent, self.env, 
                                  self.market, self.slippage).out().copy()

        if self.plot:
            utils.plot(self.bt_df)   

        self.stats(self.bt_df['Return'])
            
    def clear_session(self):
        K.clear_session()
            
class DQNAgent(Agent):
    
    def __init__(self, config, save_name=None):
        super().__init__(config, save_name)
        
    def load_model(self, model_path):
        self.model = load_model(model_path)

    def init_env(self):
        self.env = gym.make(self.env_name, csv_list=self.csv_list,
                            trading_cost=self.trading_cost, time_cost=self.time_cost, market=self.market)
        
        self.env = MyWrapper(self.env)
        
        np.random.seed(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        
        self.steps_per_episode = int(self.env.processed_array.shape[0] * self.split_ratio)
        self.nb_actions = self.env.action_space.n  
        
    def create_agent(self):
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        
        self.model = Model.simple(self.env)
        memory = SequentialMemory(limit=self.memory_len, window_length=1)
        policy = EpsGreedyQPolicy(eps=self.eps)
        self.agent = DQNAgent(model=self.model, nb_actions=self.nb_actions, memory=memory,
                              nb_steps_warmup=self.steps_per_episode,
                              target_model_update=self.target_model_update, policy=policy)
        self.agent.compile(Adam(lr=self.lr))
        
        
    def predict(self, df):
        data = self.env.fe.transform(df.iloc[:]).values
        val = np.argmax(self.agent.select_action(data[-1:], do_train=False))
        self.signal_transform(val)
    
    def signal_transform(self, val):
        if val== 0:
            print('Buy')
        elif val == 1:
            print('Sell')
        else:
            print('Close/Hold')
