from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
from sklearn import metrics

def RMSE(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 
def absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true)

svp = {'C': 1000, 
       'epsilon': 0.1,
       'gamma': 0.01, 
       'kernel': 'rbf'}
rfp = {'max_depth': 7,
       'min_samples_leaf': 2,
       'min_samples_split': 5,
       'n_estimators': 100,
       'random_state': 42}
dcp = {'criterion': 'mae', 
       'max_depth': 6,
       'min_samples_leaf': 2, 
       'min_samples_split': 3,
       'splitter':'best',
       'random_state': 42}

knp = {'algorithm': 'brute', 
       'n_neighbors': 5,
       'p': 12}

def RFSVR():
    svr = SVR(**svp)
    return svr

def KNR():
    knr = KNeighborsRegressor(**knp)
    return knr

def RFR():
    rnr=RandomForestRegressor(**rfp)
    return rnr

def DTR():
    dtr=DecisionTreeRegressor(**dcp)
    return dtr

dataFr = pd.read_csv("CFSDP.csv")
from scipy.stats import pearsonr

dataFr.head()
dataFr.shape
df = dataFr
print(df.isnull().any().any())
print(df.isnull().sum().sum())
print(df.isnull().any().any())
print(df.isnull().sum())
Xorg = df.to_numpy()  # Take one dataset: hm

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
## store these off for predictions with unseen data
Xmeans = scaler.mean_
Xstds = scaler.scale_


import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor
#from Best_parameter import *

y = Xscaled[:, 17]
X = Xscaled[:, 0:17] 


from sklearn.ensemble import RandomForestRegressor 
kf = 0
R_2 = []
mae = []
mse= []
rmse = []
score = []
n_splits = 5
rf = RandomForestRegressor()
cv_set = np.repeat(-1.,X.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)

for train_index,test_index in skf.split(X, y):
    x_train,x_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()
    tuned_parameters= {
     'max_depth': [i for i in range(2,20)],
     'min_samples_leaf': [i for i in range(2,20)],
     'min_samples_split': [2,3,5,7,9,11,13,15,17],
     'n_estimators': [10]
         }
    grid =GridSearchCV(rf,tuned_parameters,cv =5,verbose=0,scoring='neg_mean_absolute_error',n_jobs=-1)     
    grid.fit(x_train,y_train)
    best_params = grid.best_params_
    R_2.append(metrics.r2_score(y_test, grid.predict(x_test)))
    mae.append(metrics.mean_absolute_error(y_test, grid.predict(x_test)))
    mse.append(metrics.mean_squared_error(y_test, grid.predict(x_test)))
    rmse.append(RMSE(y_test, grid.predict(x_test)))
    predicted_y =  grid.predict(x_test)
    kf = kf + 1
    cv_set[test_index] = predicted_y
    print(best_params,grid.best_score_)
    
print("R^2 (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(R_2),np.std(R_2)))    
print("MAE (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(mae),np.std(mae)))    
print("MSE (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(mse),np.std(mse)))   
print("RMSE (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(rmse),np.std(rmse)))   


