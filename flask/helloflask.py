#!flask/bin/python
from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
import model


import imp
from types import ModuleType
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

munit =[]
mmodel=[]
app = Flask(__name__)

api = Api(app)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/out', methods=['GET'])
def get_tasks():
    x = np.array([1,-1,1,-1])
    pdx=pd.DataFrame(x)
    out = pdx.to_json()
    return out

@app.route('/store', methods=['GET'])
def put_tasks():
    m=request.data
    print(m)
    mBuffer = StringIO(m)
    dframe = pd.read_csv(mBuffer)
    print(dframe.head())
    print(dframe.shape)
    global munit
    print( isinstance(munit,ModuleType))
    munit.compute(dframe)
    return str(dframe.shape)

@app.route('/module', methods=['GET'])
def put_module():
    m=request.data
    with open('unit/module.py', 'w+') as myfile:
      myfile.write(m)
    global munit
    munit=imp.load_source('mymodule','unit/module.py');
    xs=str(isinstance(munit,ModuleType))
    print(xs)
    #reload('module')
   
  #  munit.compute('ABC')
    return 'OK '+xs

@app.route('/initmodel', methods=['GET'])
def initmodel():
    m=request.args.get('model')
    print('model '+m)
    global mmodel
    mmodel=model.init_model(m)
    print('type '+type(mmodel).__name__+'  '+str(type(mmodel)))
    return 'OK '+m


@app.route('/predictmodel', methods=['GET'])
def predictmodel():
    m=request.args.get('model')
    print('model '+m)
    d=request.data
    print(d)
    mBuffer = StringIO(d)
    dframe = pd.read_csv(mBuffer)
    print(dframe.head())
    print('shape '+str(dframe.shape))

    global mmodel
    print('type '+type(mmodel).__name__+'  '+str(type(mmodel)))
    if type(mmodel).__name__ =='list':
      return 'ERROR model not init',500
    
    return model.predict_model(dframe,mmodel)
 

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
    app.run(host='0.0.0.0',debug=True,port=6002)