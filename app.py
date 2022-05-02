
from flask import Flask, request
from flask import jsonify
from flask import request
from flask import abort
from json import loads
import numpy as np
from sklearn.linear_model import LinearRegression


with open('model.json', 'r') as f:
  content = f.read()
  model = loads(content)


predictor = LinearRegression(n_jobs=-1)
predictor.coef_ = np.array(model)
predictor.intercept_ = np.array([0])

app = Flask(__name__)

@app.route('/')
def hello_world():
  params = request.args.get('input')
  parameters = params.split(",")
  print(parameters[0:2])
  X_TEST = [parameters[0:1]]
  outcome = predictor.predict(X=X_TEST)
  return str(outcome[0])


if __name__ == "__main__":
  app.run()