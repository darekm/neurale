#!/usr/bin/python3
from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
import types
from computeconv import Computer

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
    return "Hello, World   from Flask!"

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
def store_data():
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
    
    global UNIT
    
    if not UNIT.was:
        print("CREATE")
    COMP.clear()   
    UNIT.CreateModel=compile_unit(m)
    UNIT.was=True
    #UNIT.model
    print('windowsize',COMP.window_size)
    cm=UNIT.CreateModel('flask',COMP.window_size)
    COMP.compute(cm)
    
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
    COMP.clear
    with open('unit/modelconv1.txt', 'r') as myfile:
        m=myfile.read()
    print(m)
    UNIT.CreateModel=compile_unit(m)
    UNIT.was=True
   
    #UNIT=imp.load_source('mymodule','unit/modelconv1.py');
    xs=str(isinstance(UNIT,ModuleType))
    print(xs)
    print('windowsize',COMP.window_size)
    COMP.compute(UNIT.CreateModel('flask',COMP.window_size))
    
    return 'OK '

@app.route('/check', methods=['GET'])
def checkmodel():
    m2=request.data
    m=m2.decode("utf-8")

    #m=request.args.get('data')
    #//    if m:
    #    print('data '+m)
    global COMP
 
    #   if not COMP.hasdata:
    #        return 'ERROR the model has no data',500
    try:
        x= COMP.predict (m)
    except Exception as ex:
        print(ex)
        return 'ERROR '+str(ex),500
    return 'OK  &'+"value= {}".format(x)
 

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
    app.run(host='0.0.0.0',debug=True,port=6005)