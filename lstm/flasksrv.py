#!/usr/bin/python3
from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
import types
from computer import Computer

import imp
from types import ModuleType
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

class A:    
    def meth1(self, par1):
        print("in A.meth1: par1 =", par1)

    
UNIT =A()
UNIT.was=False
UNIT.model=[]
mmodel=[]
app = Flask(__name__)

api = Api(app)
COMP=Computer()



def compile_unit(source):
        # compile code
        foo_code = compile(source, "<string>", "exec")
        foo_ns = {}
        exec(foo_code,foo_ns)
        return  types.MethodType(foo_ns["CreateModel"], A)


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
    with open('unit/module.txt', 'w+') as myfile:
        myfile.write(m)
  
    global COMP
    if not COMP.hasdata:
        COMP.load_dataset(mDataset)
        print('load dataset')
    
    #with open('unit/'+mName+'.py', 'w+') as myfile:
    #    myfile.write(m)
    global UNIT
    
    if not UNIT.was:
        print("CREATE")
    COMP.clear()   
    UNIT.CreateModel=compile_unit(m)
    UNIT.was=True
    #UNIT.model
    cm=UNIT.CreateModel('flask')
    #UNIT=imp.load_source('mymodule','unit/'+mName+'.py');
    #xs=str(isinstance(UNIT,ModuleType))
    #print('unit ',xs)
    #reload('module')
    #UNIT.model.compile(loss='mean_squared_error', optimizer='adam')
    #cm=UNIT.model
    #cm=UNIT.CreateModel('flask')
    COMP.compute(cm)
    #cm=UNIT.model
    
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
    x,y= COMP.run()
    return 'OK  &'+"tst= {}".format(x)+"&train= {}".format(y)
  

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