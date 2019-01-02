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

class Computer():
    """A class for an building and inferencing an lstm model"""

    def __init__(self):
        self._test = "abc"
        self.hasdata=False
        self.window_size=10
        
    def test(self):
        return self._test

    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset,
                       
                       window_size, look_ahead=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-window_size-look_ahead-1):
            a = dataset[i:(i+window_size), 0]
            b= dataset[i:(i+window_size),1]
            dataX.append((a,b))
            dataY.append(dataset[i+window_size + look_ahead-1, 0])
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
        _model.compile(loss='mean_squared_error', optimizer='adam')
        _model.fit(self.trainX, self.trainY, epochs=10, batch_size=1, verbose=1)
        self.mmodel=_model
        trainPredict = self.mmodel.predict(self.trainX)
        print(trainPredict.shape)
        
    def run(self):
        # make predictions
        if not self.mmodel:
            return "model not initialized"
        print("shapeX:",self.trainX.shape)
        trainPredict = self.mmodel.predict(self.trainX)
        testPredict = self.mmodel.predict(self.testX)
        # invert predictions
        trainYY=self.trainY.reshape(self.trainY.shape+(1,))
        testYY=self.testY.reshape(self.testY.shape+(1,))

        trainSPredict = self.inverse_dataset(trainPredict)
        trainSY = self.inverse_dataset(trainYY)
        testPredict = self.inverse_dataset(testPredict)
        testY = self.inverse_dataset(testYY)
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(self.trainY, trainPredict))
        print('Train Score: %.3f RMSE' % (trainScore))
        trainSScore = math.sqrt(mean_squared_error(trainSY, trainSPredict))
        print('Train SScore: %.3f RMSE' % (trainSScore))
        testScore = math.sqrt(mean_squared_error(testY, testPredict))
        print('Test Score: %.3f RMSE' % (testScore))
        with open('score','a') as f:
            f.write("%s test score %s  train %s\n" %
                 ( datetime.datetime.now(),testScore, trainScore))
        return testScore

    

    def load_dataset(self,model_name):
        # normalize the dataset
        P1dataset = pandas.read_csv('piec.csv', usecols=[2,3], engine='python', skipfooter=3)
   
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(P1dataset)
   
        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        print(len(train), len(test))

        # reshape into X=t and Y=t+1
        look_back = 5
        window_size=self.window_size
        self.trainX, self.trainY = self.create_dataset(train, window_size,look_back)
        self.testX, self.testY = self.create_dataset(test, window_size,look_back)
        self.hasdata=True
    
# Usage
#1. Init model to global variable
    #model = init_model("coffe_model")
#2. Run dataset_prediction(dataframe)




