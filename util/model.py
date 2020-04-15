from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate

def simple(env):
    model = Sequential()
    model.add(Flatten(input_shape=env.observation_space.shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))
    
    return model