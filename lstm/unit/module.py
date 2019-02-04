import numpy
import math
from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
from keras.layers import GRU
import keras.layers


def CreateModel(s,w):
    window_size=10
    # create and fit the LSTM network
    model = Sequential()
    model.add(GRU(4, input_shape=(2,window_size)))
    model.add(Dropout(1.5))
    model.add(Dense(1))
        #prepare
#    print('type '+type(model).__name__+'  '+str(type(mmodel)))
   
    print(s)
    return model

