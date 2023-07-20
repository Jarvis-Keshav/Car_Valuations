import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

df=pd.read_csv('Synthetic_entries.csv')
df.head()
df=df.drop(['Unnamed: 0'],axis='columns',inplace=False)

cols_drop=['Final_Price']
X=df.drop(cols_drop,axis=1)
y=df['Final_Price']
from category_encoders import TargetEncoder
te=TargetEncoder(cols=['Brand','Model','Location']).fit(X,y)
X=te.transform(X)
X.head()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
param={'n_estimators':[10,20,50,100,150,200],
 'max_depth':range(3,8),
 'eta':[0.05,0.1,0.15,0.2]}
xgr=xgb.XGBRegressor()
gs=GridSearchCV(estimator=xgr,param_grid=param,scoring='neg_root_mean_squared_error',cv=2,verbose=3)
gs.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error
xgr=xgb.XGBRegressor(eta=0.05,max_depth=7,n_estimators=200)
xgr.fit(X_train,y_train)
pred=xgr.predict(X_test)

from joblib import dump, load
dump(xgr, 'Car_Price_Predictor')

list_of_lists = []

with open('C:/Users/HarshGupta/Desktop/outcomes/detail.txt') as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(',')]
        list_of_lists.append(inner_list)

def swapPositions(list, pos1, pos2):
     
    list[0][pos1], list[0][pos2] = list[0][pos2], list[0][pos1]
    return list

pos1, pos2  = 2, 5
 
print(swapPositions(list_of_lists, pos1, pos2))

pos3, pos4  = 3, 5
print(swapPositions(list_of_lists, pos3, pos4))

pos5, pos6  = 4, 5
print(swapPositions(list_of_lists, pos5, pos6))

if list_of_lists[0][0] == 'Tvs':
    y1=float(36663.222534)
    list_of_lists[0][0]=float(y1)
elif list_of_lists[0][0] == 'Honda':
    y1=float(40627.390301)
    list_of_lists[0][0]=float(y1)
    
if list_of_lists[0][1] == 'CB_350':
    y2=float(40504.973256)
    list_of_lists[0][1]=float(y2)
elif list_of_lists[0][1] == 'Apache':
    y2=float(36959.888533)
    list_of_lists[0][1]=float(y2)
    
if list_of_lists[0][2] == 'Delhi':
    y3=float(37732.610509)
    list_of_lists[0][2]=float(y3)
elif list_of_lists[0][2] == 'Gujrat':
    y3=float(39602.542607)
    list_of_lists[0][2]=float(y3)
    
list_of_lists[0][3]=float(list_of_lists[0][3])
list_of_lists[0][4]=float(list_of_lists[0][4])
list_of_lists[0][5]=float(list_of_lists[0][5])
list_of_lists[0][6]=float(list_of_lists[0][6])
list_of_lists[0][7]=float(list_of_lists[0][7])
list_of_lists[0][8]=float(list_of_lists[0][8])

from joblib import dump, load
xgr = load('Car_Price_Predictor')

features = np.array(list_of_lists)
xgr.predict(features)

with open('C:/Users/HarshGupta/Desktop/ml_predicted.txt', 'w') as f:
    print(xgr.predict(features), file=f)
