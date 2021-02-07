#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:29:00 2020

@author: erikferrari
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

diamond = pd.read_csv("diamonds.csv")


# DATA CLEANING


diamond.columns

diamond.head()

diamond.dtypes

diamond.info()

diamond.shape

summary = diamond.describe()

#removing any duplicate values
diamond.drop_duplicates(inplace=True)

#check for null values
diamond.isnull().any()

#the min values for x, y, and z are zero, but these are dimensions and can't be zero, so we replace with NaN
diamond[['x', 'y', 'z']] = diamond[['x', 'y', 'z']].replace(0, np.NaN)

diamond.isnull().sum()
#only null values are the ones we just created created, since dataset is so large and NaN's are relatively small, we will just drop these values
diamond.dropna(inplace=True)
diamond.isnull().any()
diamond.shape

#we already have an index, no need for this column
diamond = diamond.drop(["Unnamed: 0"], axis=1)

#check for outliers
diamond.loc[diamond.price >= 18000]
#there a lot of diamonds with a price significantly higher than the mean, no need to drop 


# DATA EXPLORATION AND VISUALIZATION


#distributions of variables
plt.title('Distribution of Carat')
plt.xlabel('Carat')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['carat'], shade=True, color='r')

sns.catplot(x='cut', data=diamond, kind='count', palette="Reds")
plt.title('Distribution of Cut')

sns.catplot(x='color', data=diamond, kind='count', palette="Reds")
plt.title('Distribution of Color')

sns.catplot(x='clarity', data=diamond, kind='count', palette="Reds")
plt.title('Distribution of Clarity')

plt.title('Distribution of Depth')
plt.xlabel('Depth')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['depth'], shade=True, color='r')

plt.title('Distribution of Table')
plt.xlabel('Table')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['table'], shade=True, color='r')

plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['price'], shade=True, color='r')

plt.title('Distribution of x')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['x'], shade=True, color='r')

plt.title('Distribution of y')
plt.xlabel('y')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['y'], shade=True, color='r')

plt.title('Distribution of z')
plt.xlabel('z')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=diamond['z'], shade=True, color='r')

#relationships between all variables - checking for colinearity
plt.title('Correlation Between All Variables')
sns.heatmap(data=diamond.corr(), square=True , annot=True, cbar=True, cmap='Reds')

#relationships between features and price
plt.title('Relationship Between Carat and Price')
plt.xlabel('Carat')
plt.ylabel('Price')
sns.regplot(x='carat', y='price', data=diamond, color='r')

sns.catplot(x='cut', y='price', data=diamond, kind='box', palette='Reds')
plt.title('Relationship Between Cut and Price')

sns.catplot(x='color', y='price', data=diamond, kind='box', palette="Reds")
plt.title('Relationship Between Color and Price')

sns.catplot(x='clarity', y='price', data=diamond, kind='box', palette='Reds')
plt.title('Relationship Between Clarity and Price')

plt.title('Relationship Between Depth and Price')
plt.xlabel('Depth')
plt.ylabel('Price')
sns.regplot(x='depth', y='price', data=diamond, color='r')

plt.title('Relationship Between Table and Price')
plt.xlabel('Table')
plt.ylabel('Price')
sns.regplot(x='table', y='price', data=diamond, color='r')

plt.title('Relationship Between x and Price')
plt.xlabel('x')
plt.ylabel('Price')
sns.regplot(x='x', y='price', data=diamond, color='r')

plt.title('Relationship Between y and Price')
plt.xlabel('y')
plt.ylabel('Price')
sns.regplot(x='y', y='price', data=diamond, color='r')

plt.title('Relationship Between z and Price')
plt.xlabel('z')
plt.ylabel('Price')
sns.regplot(x='z', y='price', data=diamond, color='r')


#FEATURE ENGINEERING


#first, we add the column 'volume', which multiplies x*y*z to tell us the total size of the diamond
diamond['volume'] = diamond['x']*diamond['y']*diamond['z']

#next, we add ratio, which divides x/y to tell us the shape
diamond['ratio'] = diamond['x']/diamond['y']

#prepare data for models before processing to avoid data leakage
x = diamond.drop(['price'], axis=1) 
y = diamond.price 
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=2, test_size=0.2)

#we will create dummy variables for the categorical columns 
train_x = pd.get_dummies(train_x) 
test_x = pd.get_dummies(test_x) 
#normalize the numerical columns
diamond_norm_train = pd.DataFrame(preprocessing.normalize(train_x[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=train_x.index)
diamond_norm_test = pd.DataFrame(preprocessing.normalize(test_x[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=test_x.index)
train_x[['carat','depth','x','y','z','table']] = diamond_norm_train[['carat','depth','x','y','z','table']]
test_x[['carat','depth','x','y','z','table']] = diamond_norm_test[['carat','depth','x','y','z','table']]
#lets look at the distribution one column in the training data to see what changed
plt.title('Distribution of x')
plt.xlabel('x')
plt.ylabel('Probability Density Function')
sns.kdeplot(data=train_x['x'], shade=True, color='r')


#MODEL TRAINING


#prepare lists for model comparison
models = ['linear regression', 'lasso regression', 'ridge regression', 'decision tree', 'random forest', 'svm', 'knn']
maes = []
rmses = []
rsquares = []

#basic models

#linear regression
reg = LinearRegression()
reg.fit(train_x, train_y)
#5 fold cross-validated scores
cv_1 = cross_val_score(estimator=reg, X=train_x, y=train_y, cv=5)
#accuracy of model
score_1 = reg.score(test_x, test_y)
print(cv_1)
print('Score: ', score_1)
#predicting price values for test set
y_pred_1 = reg.predict(test_x)
#validate model
mae_1 = mean_absolute_error(test_y, y_pred_1)
rmse_1 = math.sqrt(mean_squared_error(test_y, y_pred_1))
r2_1 = r2_score(test_y, y_pred_1)
print('Mean Absolute Error:', mae_1)
print('Root Mean Squared Error:', rmse_1)
print('R^2:', r2_1)
maes.append(mae_1)
rmses.append(rmse_1)
rsquares.append(r2_1)

#lasso regression
las = Lasso()
las.fit(train_x, train_y)
#5 fold cross-validated scores
cv_2 = cross_val_score(estimator=las, X=train_x, y=train_y, cv=5)
#accuracy of model
score_2 = las.score(test_x, test_y)
print(cv_2)
print('Score: ', score_2)
#predicting price values for test set
y_pred_2 = las.predict(test_x)
#validate model
mae_2 = mean_absolute_error(test_y, y_pred_2)
rmse_2 = math.sqrt(mean_squared_error(test_y, y_pred_2))
r2_2 = r2_score(test_y, y_pred_2)
print('Mean Absolute Error:', mae_2)
print('Root Mean Squared Error:', rmse_2)
print('R^2:', r2_2)
maes.append(mae_2)
rmses.append(rmse_2)
rsquares.append(r2_2)

#ridge regression
rid = Ridge()
rid.fit(train_x, train_y)
#5 fold cross-validated scores
cv_3 = cross_val_score(estimator=rid, X=train_x, y=train_y, cv=5)
#accuracy of model
score_3 = rid.score(test_x, test_y)
print(cv_3)
print('Score: ', score_3)
#predicting price values for test set
y_pred_3 = rid.predict(test_x)
#validate model
mae_3 = mean_absolute_error(test_y, y_pred_3)
rmse_3 = math.sqrt(mean_squared_error(test_y, y_pred_3))
r2_3 = r2_score(test_y, y_pred_3)
print('Mean Absolute Error:', mae_3)
print('Root Mean Squared Error:', rmse_3)
print('R^2:', r2_3)
maes.append(mae_3)
rmses.append(rmse_3)
rsquares.append(r2_3)

#decision tree
dec = DecisionTreeRegressor(random_state=1)
dec.fit(train_x, train_y)
#5 fold cross-validated scores
cv_4 = cross_val_score(estimator=dec, X=train_x, y=train_y, cv=5)
#accuracy of model
score_4 = dec.score(test_x, test_y)
print(cv_4)
print('Score: ', score_4)
#predicting price values for test set
y_pred_4 = dec.predict(test_x)
#validate model
mae_4 = mean_absolute_error(test_y, y_pred_4)
rmse_4 = math.sqrt(mean_squared_error(test_y, y_pred_4))
r2_4 = r2_score(test_y, y_pred_4)
print('Mean Absolute Error:', mae_4)
print('Root Mean Squared Error:', rmse_4)
print('R^2:', r2_4)
maes.append(mae_4)
rmses.append(rmse_4)
rsquares.append(r2_4)

#random forest
rf = RandomForestRegressor(random_state=1)
rf.fit(train_x, train_y)
#5 fold cross-validated scores
cv_5 = cross_val_score(estimator=rf, X=train_x, y=train_y, cv=5)
#accuracy of model
score_5 = rf.score(test_x, test_y)
print(cv_5)
print('Score: ', score_5)
#predicting price values for test set
y_pred_5 = rf.predict(test_x)
#validate model
mae_5 = mean_absolute_error(test_y, y_pred_5)
rmse_5 = math.sqrt(mean_squared_error(test_y, y_pred_5))
r2_5 = r2_score(test_y, y_pred_5)
print('Mean Absolute Error:', mae_5)
print('Root Mean Squared Error:', rmse_5)
print('R^2:', r2_5)
maes.append(mae_5)
rmses.append(rmse_5)
rsquares.append(r2_5)

#let's look at feature importances
feat_importances = pd.Series(rf.feature_importances_, index=train_x.columns)
feat_importances.nlargest(20).plot(kind='barh')

#svm
svr = SVR()
svr.fit(train_x, train_y)
#5 fold cross-validated scores
cv_6 = cross_val_score(estimator=svr, X=train_x, y=train_y, cv=5)
#accuracy of model
score_6 = svr.score(test_x, test_y)
print(cv_6)
print('Score: ', score_6)
#predicting price values for test set
y_pred_6 = svr.predict(test_x)
#validate model
mae_6 = mean_absolute_error(test_y, y_pred_6)
rmse_6 = math.sqrt(mean_squared_error(test_y, y_pred_6))
r2_6 = r2_score(test_y, y_pred_6)
print('Mean Absolute Error:', mae_6)
print('Root Mean Squared Error:', rmse_6)
print('R^2:', r2_6)
maes.append(mae_6)
rmses.append(rmse_6)
rsquares.append(r2_6)

#knn
knn = KNeighborsRegressor()
knn.fit(train_x, train_y)
#5 fold cross-validated scores
cv_7 = cross_val_score(estimator=knn, X=train_x, y=train_y, cv=5)
#accuracy of model
score_7 = knn.score(test_x, test_y)
print(cv_7)
print('Score: ', score_7)
#predicting price values for test set
y_pred_7 = knn.predict(test_x)
#validate model
mae_7 = mean_absolute_error(test_y, y_pred_7)
rmse_7 = math.sqrt(mean_squared_error(test_y, y_pred_7))
r2_7 = r2_score(test_y, y_pred_7)
print('Mean Absolute Error:', mae_7)
print('Root Mean Squared Error:', rmse_7)
print('R^2:', r2_7)
maes.append(mae_7)
rmses.append(rmse_7)
rsquares.append(r2_7)

#compare models
compare = pd.DataFrame({'Algorithms' : models , 'MAEs' : maes, 'RMSEs': rmses, 'R2' : rsquares})
compare.sort_values(by='MAEs')
#will base model selection on MAE

#most accurate model is randomforest, let's optimize the hyper-parameters 
param_grid = {'bootstrap': [True,False], 'max_depth': [5,6,7], 'max_features': ['auto','sqrt','log2'], 'min_samples_leaf': [4,5,6], 'min_samples_split': [2,3,4], 'n_estimators': [6,7,8], 'n_jobs': [-1,1]}
cv_rf = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, scoring='neg_mean_absolute_error')
cv_rf.fit(train_x, train_y)
cv_rf.best_params_
cv_rf_pred = cv_rf.predict(test_x)
mae_cv_rf = mean_absolute_error(test_y, cv_rf_pred)
rmse_cv_rf = math.sqrt(mean_squared_error(test_y, cv_rf_pred))
r2_cv_rf = r2_score(test_y, cv_rf_pred)
print('Mean Absolute Error:', mae_cv_rf)
print('Root Mean Squared Error:', rmse_cv_rf)
print('R^2:', r2_cv_rf)
















