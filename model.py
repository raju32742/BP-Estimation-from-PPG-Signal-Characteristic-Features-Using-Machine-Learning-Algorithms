import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, AdaBoostRegressor
def RMSE(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())
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
import math
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from keras.models import load_model 
from sklearn import ensemble
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns;
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import seaborn as sns;
#from utils import *
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
from sklearn.neural_network import MLPRegressor
from scipy import stats
from scipy.special import boxcox, inv_boxcox
from sklearn import linear_model
import xgboost 


def RMSE(targets,predictions):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

def ME(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true - y_pred)

def absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs((y_true - y_pred) / y_true)



dataFr = pd.read_csv("ReliefMAP.csv")
dataFr.head()
dataFr.shape
df = dataFr
# df = df.drop(df.columns[48], axis=1) 

Xorg = df.to_numpy()  # Take one dataset: hm
scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorg)
Xmeans = scaler.mean_
Xstds = scaler.scale_



# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))

# Xorg = scaler.fit_transform(Xorg)

y = Xscaled[:, 16]
X = Xscaled[:,0:16]




from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers
from sklearn.metrics import r2_score
from keras.wrappers.scikit_learn import KerasRegressor

def DNN(X):
    model = Sequential()
    # The Input Layer :
    model.add(Dense(70, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
    # The Hidden Layers :
    # model.add(Dense(50, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(100, kernel_initializer='normal',activation='relu'))
    model.add(Dense(150, kernel_initializer='normal',activation='relu'))
    model.add(Dropout(0.5))
    # The Output Layer :
    model.add(Dense(1, kernel_initializer='normal',activation='linear'))
    # Compile the network :
#    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MAE'])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mse'])
    return model

grid = DNN(X)
grid.summary()



kf = 0
R_2 = []
mae = []
mse= []
rmse = []
mape= []
score = []
n_splits = 10
cv_set = np.repeat(-1.,X.shape[0])
ape = np.repeat(-1.,X.shape[0])
skf = KFold(n_splits = n_splits ,shuffle=True, random_state=42)

i = 0
for train_index,test_index in skf.split(X, y):
    x_train,x_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    if x_train.shape[0] != y_train.shape[0]:
        raise Exception()   
    grid.fit(x_train, y_train, batch_size = 32, epochs = 100)
    predicted_y =  grid.predict(x_test)
    R_2.append(metrics.r2_score(y_test, predicted_y))
    y_test = (y_test * Xstds[16]) + Xmeans[16]
    predicted_y = (predicted_y * Xstds[16]) + Xmeans[16]
    mae.append(metrics.mean_absolute_error(y_test, predicted_y))
    mse.append(metrics.mean_squared_error(y_test,predicted_y))
    rmse.append(RMSE(y_test, grid.predict(x_test)))
    mape.append(mean_absolute_percentage_error(y_test, predicted_y))
    kf = kf + 1
    output= predicted_y[:,0]
    cv_set[test_index] = output


print("R^2 (Avg. +/- Std.) is  %0.3f +/- %0.3f" %(np.mean(R_2),np.std(R_2)))    
print("MAE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mae),np.std(mae)))    
print("MSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mse),np.std(mse)))   
print("RMSE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.sqrt(np.mean(mse)),np.sqrt(np.std(mse))))
print("MAPE (Avg. +/- Std.) is  %3.3f +/- %3.3f" %(np.mean(mape),np.std(mape)))



yy = (y * Xstds[16]) + Xmeans[16]
cv_sety = cv_set 


print("Overall R^2: " + str(metrics.r2_score(yy,cv_sety)))
print("Overall MAE: " + str(metrics.mean_absolute_error(yy,cv_sety)))
print("Overall MSE: " + str(metrics.mean_squared_error(yy , cv_sety)))
print("Overall RMSE: " + str(RMSE(yy, cv_sety)))
print("Overall ME: " + str(ME(yy, cv_sety)))


me = ME(yy, cv_sety)
sd = np.sqrt(((yy - cv_sety-me) ** 2).mean())
             
sum = 0

for i in range(218):
    ii = (yy[i] - cv_sety[i]-me) ** 2
    sum = sum + ii        
sd = np.sqrt(sum/217)

me = (yy - cv_sety)

from statistics import stdev
from fractions import Fraction as fr
print("The Standard Deviation %s" %(stdev(cv_sety)))

# 0.951
# -0.533
#  4.175


# for i in me: 
#     print("%3.3f"%(i))

#     
    

# plt.figure(figsize=(4,3))
# plt.hist(x=Xorg[:,51],  bins=15, color='#0504aa', alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.6)
# plt.grid(axis='x', alpha=0.6)
# plt.xlabel("Value (mmHg)",fontsize=10)
# plt.ylabel("Frequency",fontsize=10)
# plt.savefig("imgs/"+ "MAP_Database_Hist"+".pdf", bbox_inches='tight', dpi=320)  
# plt.show()
    


    
len(me)
me = me[~(me > 5.00)]
len(me)




# SBP = 199 216
# MAP 214   217
#DBP 214 217

# # ################# ====== plot bland_altman_plot
# from graph import*
# bland_altman_plot(yy, cv_sety, "B_DBP")
# act_pred_plot(yy, cv_sety,0.951,mae,"R2_DBP")