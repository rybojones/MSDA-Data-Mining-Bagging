#!/usr/bin/env python
# coding: utf-8


'''
bagging.py: Compare bagging for linear regression and decision tree regression.
'''


__author__ = 'Ryan Jones'
__copyright__ = "Copyright March 2020, UCF MSDA Program"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def load_file(filepath):
    '''
    Load the csv file into a dataframe
    '''
    return pd.read_csv(filepath)


def split_X_and_y(dataframe):
    '''
    Returns X dataframe and y series representing predictors and response sets, respectively.
    Note: Assumes the response feature is in the last column.
    '''
    return dataframe.iloc[:,:-1], dataframe.iloc[:,-1]


def bootstrap_sample(dataframe, sample_size=1460):
    '''
    Perform bootstrap sampling on the dataframe, i.e. sampling with replacement
    '''
    # even though the sampling is with replacement, it seems practical to limit the max sample size to be the number of data records.
    if X.shape[0] < sample_size:
        print('Error: Sample size out of bounds.')
        return

    # randomly sample "sample_size" values with replacement from "dataframe"
    return dataframe.sample(n=sample_size, replace=True, axis=0)


def regressor(dataframe, estimator=LinearRegression(), n_estimators=20):
    '''
    Perform regression on the dataframe (as defined by "estimator") n_estimator times. Return dataframe of predictions for each iteration.
    '''
    # initialize an empty dataframe to store predicted values
    predictions = pd.DataFrame()

    # perform regression with bagging and store the predictions
    for i in range(n_estimators):
        df_temp = bootstrap_sample(dataframe)
        X, y = split_X_and_y(df_temp)
        reg = estimator.fit(X,y)
        predict = reg.predict(X)
        predictions['Estimator_'+str(i+1)] = predict

    return predictions


def bagging_estimator(df_predict, take_avg=True):
    '''
    Takes in a dataframe of predictions and averages those values to produce a single bagging estimator.

    'take_avg' value of True indicates the use of the arithmetic mean, False indicates the use of mode.
    '''
    # find the 'width' of the prediction dataframe
    width = df_predict.shape[1]

    # find the average of the predictions to produce a bagging estimate for linear regression
    if take_avg:
        return (df_predict.sum(axis=1) / width)

    # find the mode of the predictions to produce a bagging estimate for decision tree regression
    else:
        return df_predict.mode(axis=1).iloc[:,0]    # grab the element in the first column from the resulting dataframe to handle duplicate mode values


def mse_calculator(predict, actual):
    '''
    Output the MSE of the predicted vs. actual values.
    '''
    # check that the arrays are the same length
    if predict.shape[0] != actual.shape[0]:
        print('Error: predict and actual arrays not the same shape.')

    # define the number of data values
    n = predict.shape[0]

    # calcualte and return MSE
    sse = (predict - actual)**2
    sum_sse = sse.sum()
    return sum_sse / n
