import numpy
import matplotlib.pyplot as plt
import pandas
import math
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as Kback
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing 
from sklearn import utils

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
    def inverse_dataset(self,dset):
        x=numpy.append(dset,dset,1)
        yy= self.scaler.inverse_transform(x)
        return yy[:,0]
    
    def clear(self):
        Kback.clear_session()
  
 
    def compute (self,_model):
        print("shapeX:",self.trainX.shape)
        print("shapeY:",self.trainYO.shape)
        
        _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        _model.fit(self.trainX, self.trainYO, epochs=20, batch_size=16, verbose=2)
        score = _model.evaluate(self.testX,self.testYO, verbose=1) 
        #print('loss:', score[0])
        self.mmodel=_model
        mx1=self.testX[0:1,:]
        print('mx1',mx1.shape)
        mp1=self.mmodel.predict(mx1)
        print(mp1)
        #trainPredict = self.mmodel.predict(self.trainX)

        
    def predict(self,_data):
        mdf=numpy.fromstring(_data, sep=',')
        mx3=mdf.reshape(1,-1)
        mx2=self.scaler.transform(mx3)
        mx1=mx2.reshape(1,48,-1)
        print('MM',mx1.shape)
        
        if not self.mmodel:
            raise Exception( "model not initialized")
        
        mp1=self.mmodel.predict(mx1)
        print('predict',mp1)
        tmp1=numpy.argmax(mp1,1)
        print('return:',tmp1)
        tmp2=self.le.inverse_transform(tmp1)
        print('inverse:',tmp2,"   ",tmp2[0])
        return tmp2[0]

        
    def run(self):
        # make predictions
        if not self.mmodel:
            return "model not initialized"
        print("shapeX:",self.testX.shape)
        print("shapeYO:",self.testYO.shape)
  
        #testPredict = self.inverse_dataset(testPredict)
        #testY = self.inverse_dataset(testYY)
        score = self.mmodel.evaluate(self.testX,self.testYO, verbose=2) 
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        
        with open('scoreconv','a') as f:
            f.write("%s loss %s  accuracy %s\n" %
                 ( datetime.datetime.now(),score[0], score[1]))
        return score[0],score[1]

    

    def load_dataset(self,model_name):
        # normalize the dataset
        csv_url='mrec20190218.csv'
        print('load dataset ',model_name,'  ',csv_url)
 
        MM= pandas.read_csv(csv_url,  engine='python', skipfooter=1)
        Mdataset=MM.sample(frac=1)
        Xdataset=Mdataset.iloc[:,2:]
        Ydataset = Mdataset['training']

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset2 = self.scaler.fit_transform(Xdataset)
        dataset=dataset2.reshape(len(dataset2),48,32)
   
   
        # split into train and test sets
        train_size = int(len(dataset) * 0.77)
        test_size = len(dataset) - train_size
        
        self.trainX = dataset[0:train_size,:]
        self.testX= dataset[train_size:len(dataset),:]
        print('trainX:',len(self.trainX), len(self.testX))

        # reshape into X=t and Y=t+1
        self.window_size=7
        
        
        self.le = preprocessing.LabelEncoder()
        YN=utils.column_or_1d(Ydataset, warn=True)
        YO=self.le.fit_transform(YN)
        self.Yclasses=len(self.le.classes_)
        self.window_size=self.Yclasses
        self.trainY = YO[0:train_size]
        self.testY=YO[train_size:len(YO)]
        print('trainY:',len(self.trainY), len(self.testY))
        self.trainYO = to_categorical(self.trainY, num_classes=self.Yclasses)
        self.testYO = to_categorical(self.testY, num_classes=self.Yclasses)


        self.hasdata=True
    




