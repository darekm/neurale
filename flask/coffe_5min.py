import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
#import scipy.fftpack
from sklearn.preprocessing import StandardScaler

from keras import backend as K

def predict_model(df,_model):
    print "shape",df["training"]
    val = prepare_data(df)
    r = str(_model.predict_generator(val,steps=1))
    print ("\nasked",r)
    #K.clear_session()
    return r

def prepare_data(df):
    #prepare
    #df = make_5m_bucks(df)
    gen = ImageDataGenerator()
    data_elems = df.drop(columns=["training","time"]).values
    #data_elems = np.array([data_elems.iloc[:,:150].values,data_elems.iloc[:,200:350].values]).reshape(1,-1)
    print "dataElems", type(data_elems)
    norm_data=[]
    for k in data_elems[:]:
        norm_data.append(StandardScaler().fit_transform(k.reshape(-1,1)).reshape(-1))
    data_elems = np.array(norm_data,dtype="float32")
    print "data_elems",data_elems.shape
#    data_elems=data_elems[:2*15*20]
    label_elems = df["training"][:1].values
    #label_elems = np.array([0,0,0,0],dtype="float32").reshape(-1,);
    #convert to generator
    data_reshaped = np.array([k.reshape((2,10,10)) for k in data_elems],dtype="float32")
    validation_generator = gen.flow(data_reshaped, label_elems,batch_size=16,shuffle=False)
    print "val\n",str(data_reshaped.shape)
    return validation_generator
#    return     data_reshaped
    

def init_model(model_name):
    print "\nhello"
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_name+'.h5')
    # compile model
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model._make_predict_function()
    return model
def make_5m_bucks(df):
    print "5m bucks",df.columns
    #df.time=df.time+60*60*1000
    a,b = df.iloc[:,1:101], df.iloc[:,101:201]#,df.iloc[:,201:301],df.iloc[:,301:401];,c,d
    x = pd.DataFrame(b.join(d).join(df.time).join(df.training))
    col = x.columns
    y = pd.DataFrame(a.join(c).join(df.time+5*60*1000).join(df.training).values,columns=col)
    return x.append(y)
# Usage
#1. Init model to global variable
    #model = init_model("coffe_model")
#2. Run dataset_prediction(dataframe)





