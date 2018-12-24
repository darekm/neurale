import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
    def inverse_dataset(dset):
        x=numpy.append(dset,dset,1)
        yy= self.scaler.inverse_transform(x)
        return yy[:,0]

    def compute (self,_model):
        _model.compile(loss='mean_squared_error', optimizer='adam')
        _model.fit(self.trainX, self.trainY, epochs=20, batch_size=1, verbose=1)
        self.mmodel=_model
  
    def run(self):
        # make predictions
        trainPredict = self.mmodel.predict(self.trainX)
        testPredict = self.mmodel.predict(self.testX)
        # invert predictions
        trainSPredict = self.inverse_dataset(trainPredict)
        trainSY = self.inverse_dataset(self.trainY)
        testPredict = self.inverse_dataset(testPredict)
        testY = self.inverse_dataset(self.testY)
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(self.trainY, self.trainPredict))
        print('Train Score: %.3f RMSE' % (trainScore))
        trainSScore = math.sqrt(mean_squared_error(self.trainSY, self.trainSPredict))
        print('Train SScore: %.3f RMSE' % (trainSScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.3f RMSE' % (testScore))
        f.open('score','a')
        f.write("%s test score %s  train %s\n" %
                 (scr, datetime.datetime.now(),testScore, trainScore))
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




