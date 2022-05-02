from json import loads
import numpy as np
from sklearn.linear_model import LinearRegression

X_TEST = [[10, 20, 30]]

with open('model.json', 'r') as f:
  content = f.read()
  model = loads(content)


predictor = LinearRegression(n_jobs=-1)
predictor.coef_ = np.array(model)
predictor.intercept_ = np.array([0])

outcome = predictor.predict(X=X_TEST)

print('Outcome : {}\nCoefficients : {}'.format(outcome, model))
