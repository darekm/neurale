#!/usr/bin/python3
from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
from computer import Computer

import imp
from types import ModuleType
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

UNIT =[]
mmodel=[]
app = Flask(__name__)

api = Api(app)
COMP=Computer()

@app.route('/')
def index():
    return "Hello, World   from LSTM!"

@app.route('/test')
def test():
    global COMP
    return COMP.test()


@app.route('/out', methods=['GET'])
def get_tasks():
    x = np.array([1,-1,1,-1])
    pdx=pd.DataFrame(x)
    out = pdx.to_json()
    return out

@app.route('/store', methods=['GET'])
def put_tasks():
    m=request.data
#    print(m)
    mBuffer = StringIO(m)
    dframe = pd.read_csv(mBuffer)
    #print(dframe.head())
    print(dframe.shape)
    global UNIT
    print( isinstance(UNIT,ModuleType))
    dresult=UNIT.compute(dframe)
    dresult.to_csv('mrecout.csv', index=False, header=True)
    #return 'mrecout.csv' 
    print( dresult.groupby('labels').count())
    return str(dresult.groupby('labels').count())

@app.route('/module', methods=['GET'])
def put_module():
    mDataset=request.args.get('dataset')
    mName=request.args.get('model')
    m2=request.data
    m=m2.decode("utf-8")
    print(mName)
    global COMP
    if not COMP.hasdata:
        COMP.load_dataset(mDataset)
    
    with open('unit/'+mName+'.py', 'w+') as myfile:
        myfile.write(m)
    global UNIT
    UNIT=imp.load_source('mymodule','unit/'+mName+'.py');
    xs=str(isinstance(UNIT,ModuleType))
    print('unit ',xs)
    #reload('module')
   
   
    global COMP
    COMP.compute(UNIT.CreateModel('flask'))
    
  #  munit.compute('ABC')
    return 'OK '


@app.route('/start', methods=['GET'])
def start_module():
    m=request.args.get('dataset')
   
    global COMP
    if not COMP.hasdata:
        COMP.load_dataset(m)
        
    if not COMP.hasdata:
        return 'ERROR the model has no data',500
    global UNIT
    UNIT=imp.load_source('mymodule','unit/module.py');
    xs=str(isinstance(UNIT,ModuleType))
    print(xs)
    COMP.compute(UNIT.CreateModel('flask',COMP.window_size))
    
  #  munit.compute('ABC')
    return 'OK '


@app.route('/run', methods=['GET'])
def runmodel():
    m=request.args.get('model')
    if m:
        print('model '+m)

    global COMP
 
    if not COMP.hasdata:
        return 'ERROR the model has no data',500
    x= COMP.run()
    return 'OK '+"tst= {}".format(x)
  

@app.route('/preparedata', methods=['GET'])
def preparedata():
    m=request.args.get('dataset')
    if m:
        print('dataset '+m)
    
    global COMP
    COMP.load_dataset(m)
    
    return 'OK'
 

@app.route('/mrec', methods=['GET'])
def get_file():
    with open('mrecout.csv', 'r') as myfile:
        data=myfile.read()
        return data
    
    
class TaskListAPI(Resource):
    def get(self):
        pass

    def post(self):
        pass


if __name__ == '__main__':
    api.add_resource(TaskListAPI, '/tasks', endpoint = 'tasks')
    app.run(host='0.0.0.0',debug=True,port=6004)