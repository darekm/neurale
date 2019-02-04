import numpy
import matplotlib.pyplot as plt
import pandas
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as Kback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils import to_categorical

class Computer():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self._test = "abc"
        self.hasdata=False
        self.window_size=60
        
    def test(self):
        return self._test

    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset,               
                       window_size):
        dataX, dataY = [], []
        for i in range(len(dataset)-window_size-1):
            a = dataset[i:(i+window_size), 0]
            dataX.append(a)
            n=((i / 6) % 24)
            #n=((i+look_back) / 6) % 24
            #n=n/24
            dataY.append(n)
        return numpy.array(dataX), numpy.array(dataY)

    # convert an array of values into a dataset matrix
    def inverse_dataset(self,dset):
        x=numpy.append(dset,dset,1)
        yy= self.scaler.inverse_transform(x)
        return yy[:,0]
    
    def clear(self):
        Kback.clear_session()
  
 
    def compute (self,_model):
        print("shapeX:",self.trainX.shape)
        print("shapeY:",self.trainY.shape)
        _model.compile(loss='mean_squared_error', optimizer='rmsprop')
        _model.fit(self.trainX, self.trainYO, epochs=10, batch_size=1, verbose=1)
        self.mmodel=_model
        trainPredict = self.mmodel.predict(self.trainX)
        print(trainPredict.shape)
        
    def run(self):
        # make predictions
        if not self.mmodel:
            return "model not initialized"
        print("shapeX:",self.trainX.shape)
        trainPredict1 = self.mmodel.predict(self.trainX)
        testPredict1 = self.mmodel.predict(self.testX)        
        
        trainPredict=numpy.argmax(trainPredict1,1)
        testPredict=numpy.argmax(testPredict1,1)

        #testPredict = self.inverse_dataset(testPredict)
        #testY = self.inverse_dataset(testYY)
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(self.trainY, trainPredict))
        print('Train Score: %.3f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.testY, testPredict))
        print('Test Score: %.3f RMSE' % (testScore))
        
        with open('scorehour','a') as f:
            f.write("%s test score %s  train %s\n" %
                 ( datetime.datetime.now(),testScore, trainScore))
        return testScore

    

    def load_dataset(self,model_name):
        # normalize the dataset
        P1dataset = pandas.read_csv('piec.csv', usecols=[2,3], engine='python', skipfooter=3)
   
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(P1dataset)
   
        # split into train and test sets
        train_size = int(len(dataset) * 0.77)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))

        # reshape into X=t and Y=t+1
        window_size=self.window_size
        self.trainX, self.trainY = self.create_dataset(train, window_size)
        self.testX, self.testY = self.create_dataset(test, window_size)
        self.trainYO = to_categorical(self.trainY, num_classes=25)
        self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        self.hasdata=True
    




