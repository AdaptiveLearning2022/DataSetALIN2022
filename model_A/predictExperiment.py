#coding=UTF-8

import pdb
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn import preprocessing
import pandas as pd
import torch
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor as DNN
from sklearn.metrics import r2_score

#student_data_processed.csv
#4966 rows, 46 features

print('modeling,Data Science 2022-9-19')
df = pd.read_csv('student_data_processed.csv')
# pdb.set_trace()
df0 = df.iloc[:,1:-1]
df0=shuffle(df0)

#supervised information, -label, which column = averagedCorrectness
# remove cheating columns, dependency, which columns = ? remove maxCorrectness, minCorrectness, totalNumProblems

# swap y and z for different prediction target 
y = df0['averagedCorrectness']
z = df0['averagedTimespent']     # 

df0.drop('studentID', 1, inplace=True)
df0.drop('averagedCorrectness', 1, inplace=True)
df0.drop('averagedTimespent', 1, inplace=True)
x=df0

# model 1 XGBoost

num_train_size = 3000 # out of 4900
# to array
train_set_y=np.array(y).astype(float)
train_set_feature = np.array(x).astype(float)
print('get train set y, feature')
# pdb.set_trace()

# normalization values
# normalise across all data
feature=train_set_feature
feature=preprocessing.MinMaxScaler().fit_transform(feature)
train_set_feature_norm=feature[0:num_train_size,:]
test_set_feature_norm=feature[num_train_size:-1,:]
test_set_y = train_set_y[num_train_size:-1]
train_set_y = train_set_y[0:num_train_size]
print('xgb -- normalization')
# pdb.set_trace()

#handle Nan
train_set_feature_norm = np.nan_to_num(train_set_feature_norm)
test_set_feature_norm = np.nan_to_num(test_set_feature_norm)

x_train=train_set_feature_norm
y_train=train_set_y
x_test=test_set_feature_norm
y_test=test_set_y

print('xgb -- model')
# pdb.set_trace()

#model = XGBClassifier()
model=XGBRegressor()
model.fit(x_train,y_train)
score = model.score(x_train,y_train)
print('training set-----XGBRegressor')
print(score)
score = model.score(x_test,y_test)
print('test set-----XGBRegressor')
print(score)


# pdb.set_trace()

# model 2 LightGBM
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(x_train, y_train,eval_set=[(x_test, y_test)],eval_metric='l1',early_stopping_rounds=5)
score_gbm_train = gbm.score(x_train, y_train)
print('training set-------LightGBM')
print(score_gbm_train)
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)
score_gbm_test = gbm.score(x_test,y_test)
print('test set-------LightGBM')
print(score_gbm_test)




# model 3 RandomForest
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x_train, y_train)
score_rf_train = rf.score(x_train, y_train)
print('training set-------RF')
print(score_rf_train)
score_rf_test = rf.score(x_test,y_test)
print('test set-------RF')
print(score_rf_test)


# model 3 DNN
dnn = DNN(hidden_layer_sizes=(10,), max_iter=2500, random_state=420)
dnn.fit(x_train, y_train)
score_dnn_train = dnn.score(x_train, y_train)
print('training set-------DNN')
print(score_dnn_train)
score_dnn_test = dnn.score(x_test,y_test)
print('test set-------DNN')
print(score_dnn_test)




# pdb.set_trace()
