import pandas as pd
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def CreateModel(s,window_size):
    
    model = Sequential()
    model.add(LSTM(4, input_shape=(2,window_size)))
    model.add(Dense(1))

    print(s)
    return model
