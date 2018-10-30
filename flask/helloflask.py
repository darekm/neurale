#!flask/bin/python
from flask import Flask
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/out', methods=['GET'])
def get_tasks():
    x = np.array([1,-1,1,-1])
    pdx=pd.DataFrame(x)
    out = pdx.to_json()
    return out

@app.route('/mrec', methods=['GET'])
def get_file():
    with open('mrecout.csv', 'r') as myfile:
        data=myfile.read()
        return data

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=6002)