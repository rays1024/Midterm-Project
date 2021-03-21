# DATA 410 Advanced Applied Machine Learning Midterm Project
In this project, we will perform regularized regressions, Stepwise Regression, kernel regressions, Random Forest, XGBoost algorithm, and Deep Learning method in analyzing the Boston Housing Price dataset. The regularized regressions include Ridge, LASSO, Elastic Net, SCAD, and Square Root LASSO regression. The kernel regressions include Gaussian, Tricubic, Quartic, and Epanechnikov kernels. In hyperparameter tuning process, we will use the Particle Swarm Optimization where applicable. At the end, we will list the 5-Fold validated mean absolute values of the differences between predicted housing prices and the test group housing prices and compare the performance of each technique that we have applied in this project.

## General Imports
These imports are the tools for regularization techniques, hyperparameter tuning, and 5-Fold validation process.
```python
pip install pyswarms
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

```python
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pyswarms as ps
from numba import jit, prange
import statsmodels.api as sm
import matplotlib as mpl
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
```

## Data

We used the Boston Housing Price dataset to compare different regularization techniques. The x variables include crime, rooms, residential, industrial, nox, older, distance, highway, tax, ptratio, lstat. We used cmedv as the y variable. Then we created training and testing groups for the 5-Fold Validation process.

```python
df = pd.read_csv('/content/Boston Housing Prices.csv')
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
X = np.array(df[features])
y = np.array(df['cmedv']).reshape(-1,1)
Xdf = df[features]
kf = KFold(n_splits=5,shuffle=True,random_state=1693)

L_test = []
L_train = []
for idxtrain, idxtest in kf.split(X):
  L_test.append(idxtest)
  L_train.append(idxtrain)
```


## Ridge Regularization
The Ridge Regression is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}\beta_i^2" title="\text{minimize}\, \frac{1}{n}\text{SSR} + K\sum\limits_{i=1}^{n}\beta_i^2" /></a>

where SSR is the squared residual and K is a tuning parameter.

We used the "statsmodels.api" package to calculate the 5Fold validated MAE. Since the Elastic Net technique is the combination of Ridge and LASSO, we can set the L1 weight to be 0 to obtain Ridge regression results. We wrote a user-defined function using the "just-in-time" Numba complier to find the best hyperparameter and calculate the 5-fold validated MAE.

```python
@jit
def RidgeCV(alpha):
  PEV = []
  for i in range(len(alpha[:,0])):
    PE = []
    a = alpha[i,0]
    for i in range(5):
      idxtrain = L_train[i]
      idxtest = L_test[i]
      X_train = X[idxtrain,:]
      y_train = y[idxtrain]
      X_test = X[idxtest,:]
      y_test = y[idxtest]
      model = sm.OLS(y_train,X_train)
      result = model.fit_regularized(method='elastic_net', alpha=a, L1_wt=0)
      yhat_test = result.predict(X_test)
      PE.append(MAE(y_test,yhat_test))
    PEV.append(np.mean(PE))
  return np.array(PEV)
```

Here we set the lower bound of the hyperparameter for alpha values in Ridge Regression to be 0.001 and upper bound to be 10. We used 10 particles and 100 iterations for the particle swarm optimization process.
```python
x_max = 10 *np.ones(1)
x_min = 0.001 * np.ones(1)
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds)

cost, pos = optimizer.optimize(RidgeCV, iters=100)
```
best cost: 3.5022712055902105, best pos: [0.00682889]

This output means that the lowest MAE is $3502.27 and the best alpha value is 0.006828889.

## Least Absolute Shrinkage and Selection Operator (LASSO)
The LASSO Regression is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}|\beta_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\sum\limits_{i=1}^{n}|\beta_i|" title="\text{minimize} \frac{1}{n}\text{SSR} + K\sum\limits_{i=1}^{n}|\beta_i|" /></a>

where SSR is the squared residual and K is a tuning parameter.

We used the "statsmodels.api" package and its upgrade from GitHub to calculate the KFold validated MAE. Since the Elastic Net technique is the combination of Ridge and LASSO, we can set the L1 weight to be 1 to obtain LASSO regression results. We wrote a user-defined function using the "just-in-time" Numba complier to find the best hyperparameter and calculate the 5-fold validated MAE.

```python
@jit
def LassoCV(alpha):
  PEV = []
  for i in range(len(alpha[:,0])):
    PE = []
    a = alpha[i,0]
    for i in range(5):
      idxtrain = L_train[i]
      idxtest = L_test[i]
      X_train = X[idxtrain,:]
      y_train = y[idxtrain]
      X_test = X[idxtest,:]
      y_test = y[idxtest]
      model = sm.OLS(y_train,X_train)
      result = model.fit_regularized(method='elastic_net', alpha=a, L1_wt=1)
      yhat_test = result.predict(X_test)
      PE.append(MAE(y_test,yhat_test))
    PEV.append(np.mean(PE))
  return np.array(PEV)
```

Here we set the lower bound of the hyperparameter for alpha values in LASSO Regression to be 0.001 and upper bound to be 10. We used 10 particles and 100 iterations for the particle swarm optimization process.
```python
x_max = 10 *np.ones(1)
x_min = 0.001 * np.ones(1)
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds)

cost, pos,  = optimizer.optimize(LassoCV, iters=100)
```
best cost: 3.5991855957545704, best pos: [0.12359355]

This output means that the lowest MAE is $3599.19 and the best alpha value is 0.12359355.


## Elastic Net
The Elastic Net is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|&plus;(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{minimize}\,&space;\frac{1}{n}\text{SSR}&space;&plus;&space;K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|&plus;(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" title="\text{minimize}\, \frac{1}{n}\text{SSR} + K\left(\alpha\sum\limits_{i=1}^{n}|\beta_i|+(1-\alpha)\sum\limits_{i=1}^{n}\beta_i^2\right)" /></a>

where SSR is the squared residual, K is a tuning parameter, and a is the weight for Ridge and LASSO regression.

We wrote a user-defined function using the "just-in-time" Numba complier to find the best hyperparameters and calculate the 5-fold validated MAE. Since there are two hyperparameters in Elastic Net regression, we have two looping process to find the best combinations of the two hyperparameters.

```python
@jit
def ElasticNetCV(hyper):
  PEV = []
  for i in range(len(hyper[:,0])):
    PE = []
    a = hyper[i,0]
    for j in range(len(hyper[:,1])):
      w = hyper[j,1]
      for i in range(5):
        idxtrain = L_train[i]
        idxtest = L_test[i]
        X_train = X[idxtrain,:]
        y_train = y[idxtrain]
        X_test = X[idxtest,:]
        y_test = y[idxtest]
        model = sm.OLS(y_train,X_train)
        result = model.fit_regularized(method='elastic_net', alpha=a, L1_wt=w)
        yhat_test = result.predict(X_test)
        PE.append(MAE(y_test,yhat_test))
    PEV.append(np.mean(PE))
  return np.array(PEV)
```

Here we set the lower bounds of the hyperparameters for alpha and L1 weight values in Elastic Net Regression to be 0.001 and upper bounds to be 5 and 1 respectively. Since we have two hyperparameters, our search dimension parameter is 2. We used 10 particles and 10 iterations for the particle swarm optimization process.
```python
x_max = [5,1] *np.ones(2)
x_min = 0.001 * np.ones(2)
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

cost, pos = optimizer.optimize(ElasticNetCV, iters=10)
```
best cost: 3.6111177463331705, best pos: [0.05282937 0.76712692]

This output means that the lowest MAE is $3611.12, the best alpha value is 0.05282937, and the best L1 weight is 0.76712692.


## Square Root LASSO
The Square Root LASSO is given by the following formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=\displaystyle\text{minimize}&space;\sqrt{\frac{1}{n}\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2}&space;&plus;\alpha\sum\limits_{i=1}^{p}|\beta_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\displaystyle\text{minimize}&space;\sqrt{\frac{1}{n}\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2}&space;&plus;\alpha\sum\limits_{i=1}^{p}|\beta_i|" title="\displaystyle\text{minimize} \sqrt{\frac{1}{n}\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2} +\alpha\sum\limits_{i=1}^{p}|\beta_i|" /></a>

It minimizes the average of square root SSR with a L1 penalty function. We wrote a user-defined function using the "just-in-time" Numba complier to find the best hyperparameter and calculate the 5-fold validated MAE.

```python
@jit
def SqrtLassoCV(alpha):
  PEV = []
  for i in range(len(alpha[:,0])):
    PE = []
    a = alpha[i,0]
    for i in range(5):
      idxtrain = L_train[i]
      idxtest = L_test[i]
      X_train = X[idxtrain,:]
      y_train = y[idxtrain]
      X_test = X[idxtest,:]
      y_test = y[idxtest]
      model = sm.OLS(y_train,X_train)
      result = model.fit_regularized(method='sqrt_lasso', alpha=a)
      yhat_test = result.predict(X_test)
      PE.append(MAE(y_test,yhat_test))
    PEV.append(np.mean(PE))
  return np.array(PEV)
``` 

Here we set the lower bound of the hyperparameters for alpha in Square Root LASSO Regression to be 0.001 and upper bound to be 10. We used 10 particles and 100 iterations for the particle swarm optimization process.
```python
x_max = 10 *np.ones(1)
x_min = 0.001 * np.ones(1)
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options, bounds=bounds)

cost, pos = optimizer.optimize(SqrtLassoCV, iters=100)
```
best cost: 3.5022712055902105, best pos: [0.00682889]

This output means that the lowest MAE is $3502.27, the best alpha value is 0.00682889.


## Smoothly Clipped Absolute Deviation (SCAD)
The SCAD was introduced by Fan and Li to make penalties for large values of betas constant and not penalize more as beta values increase. A part of codes we used was from https://andrewcharlesjones.github.io/posts/2020/03/scad/. 
```python
@jit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```
```python
def scad_model(X,y,lam,a):
  n = X.shape[0]
  p = X.shape[1]
  X = np.c_[np.ones((n,1)),X]
  def scad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
  def dscad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  b0 = np.ones((p+1,1))
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
  return output.x
```
```python
def scad_predict(X,y,lam,a):
  beta_scad = scad_model(X,y,lam,a)
  n = X.shape[0]
  p = X.shape[1]
  X = np.c_[np.ones((n,1)),X]
  return X.dot(beta_scad)
```

We wrote a user-defined function using the "just-in-time" Numba complier to find the best hyperparameter and calculate the 5-fold validated MAE.
```python
@jit
def ScadCV(hparam):
  PEV = []
  for i in prange(len(hparam[:,0])):
    for j in prange(len(hparam[:,1])):
      PE  = []
      for i in range(5):
        idxtrain = L_train[i]
        idxtest = L_test[i]
        X_train = X[idxtrain,:]
        y_train = y[idxtrain]
        X_test  = X[idxtest,:]
        y_test  = y[idxtest]
        beta_scad = scad_model(X_train,y_train,hparam[i,0],hparam[j,1])
        n = X_test.shape[0]
        p = X_test.shape[1]
        # we add an extra columns of 1 for the intercept
        X1_test = np.c_[np.ones((n,1)),X_test]
        yhat_scad = X1_test.dot(beta_scad)
        PE.append(MAE(y_test,yhat_scad))
    PEV.append(np.mean(PE))
  return np.array(PEV)
```

Here we set the lower bounds of the hyperparameters for lambda and a in SCAD to be 0.001 and 1.001, and upper bounds to be 2. Since there are two hyperparameters in SCAD, our search dimension is 2. We used 20 particles and 20 iterations for the particle swarm optimization process.
```python
x_max = np.array([2, 2])
x_min = np.array([0.001, 1.0001])
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=bounds)

cost, pos = optimizer.optimize(ScadCV, iters=20)
```
3.619469998140474, best pos: [1.09053869 1.94757987]

This output means that the lowest MAE is $3619.47, the best lambda value is 1.09053869, and the best a value is 1.94757987.




## Technique Rankings
