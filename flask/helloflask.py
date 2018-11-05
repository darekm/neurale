#!flask/bin/python
from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pandas as pd
import imp
from imp import reload 


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
    x = np.array([1,-1,1,-1])
    pdx=pd.DataFrame(x)
    out = pdx.to_json()
    return out

@app.route('/module', methods=['GET'])
def put_module():
    m=request.data
    with open('unit/module.py', 'w+') as myfile:
      myfile.write(m)
    munit=imp.load_source('mymodule','unit/module.py');
    #reload('module')
   
    munit.compute('ABC')
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
    app.run(host='0.0.0.0',debug=True,port=6002)