#!/usr/bin/env python

import numpy as np
import pandas as pd
from pandas import read_csv
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

DATA_INPUT_DIR = "../../data/input"
DATA_OUTPUT_DIR = "../../data/output"
TIME_VARIABLES = ['day', 'hour', 'month', 'dayofweek', 'quarter', 'date', 'time']
VARS_TO_STANDARDIZE = ['temp', 'atemp', 'humidity', 'windspeed']
POISSON_FEATURES = ['const', 'hour', 'atemp', 'month', 'humidity', 'holiday', 'weather']
LR_FEATURES = ['hour', 'month', 'atemp_zscore', 'humidity_zscore','windspeed_zscore', 'weather']
RIDGE_FEATURES = ['dayofweek', 'hour', 'month', 'season', 'holiday', 'workingday', 'weather'
            , 'atemp_zscore', 'humidity_zscore', 'windspeed_zscore']
RF_FEATURES = ['dayofweek', 'hour', 'month', 'season', 'holiday', 'workingday', 'weather'
            , 'atemp_zscore', 'humidity_zscore', 'windspeed_zscore']
LASSO_FEATURES = ['dayofweek', 'hour', 'month', 'season', 'holiday', 'workingday', 'weather'
            , 'atemp_zscore', 'humidity_zscore', 'windspeed_zscore']
FEATURES = ['dayofweek', 'hour', 'month', 'season', 'holiday', 'workingday', 'weather', 'temp'
            , 'atemp', 'humidity', 'windspeed']
TARGET = ['count']
DATETIME_COLNAME = 'datetime'
MAX_ITER = 1000
TEST_SAMPLE_PERCENTAGE = .3
RANDOM_STATE = 0

filename_train = DATA_INPUT_DIR + "/train.csv"
filename_test = DATA_INPUT_DIR + "/test.csv"
outfile_poisson_preds = DATA_OUTPUT_DIR + "/poisson_predict.csv"
outfile_lr_preds = DATA_OUTPUT_DIR + "/lr_predict.csv"
outfile_ridge_preds = DATA_OUTPUT_DIR + "/ridge_predict.csv"
outfile_lasso_preds = DATA_OUTPUT_DIR + "/lasso_predict.csv"
outfile_rf_preds = DATA_OUTPUT_DIR + "/rf_predict.csv"


def set_time_variables(dat, time_variables):
    """Sets the time variables in a pandas dataframe
    @param dat A pandas dataframe with an index of type DatetimeIndex
    @param time_variables A list of strings representing a time metric
    @return A pandas dataframe with added columns of time metrics
    """
    for i in time_variables:
        dat[i] = getattr(dat.index, i)
    return dat

def append_standardized_vars(dat, variables):
    """Append standardized variables to a pandas dataframe
    @param dat A pandas dataframe
    @param variables A list of variable names to create standardized columns of
    @return A pandas dataframe with the newly appended variables
    """
    for variable in variables:
        variable_zscore = variable + '_zscore'
        dat[variable_zscore] = (
            dat[variable] - dat[variable].mean())/dat[variable].std()
    return dat

def read_in_data(filename, datetime_colname):
    """Read data intp pandas dataframe - must contain a datetime column
    @param filename The filename of the data
    @param datetime_colname The column name representing the datetime variable
    @return A pandas dataframe
    """
    return pd.read_csv(filename, parse_dates=datetime_colname, index_col=datetime_colname)

def prepare_data(dat, time_variables, vars_to_standardize):
    """Prepares data by setting the time variables and getting standardized values
    @param dat A pandas df
    @param time_variables A list of time variables to create
    @param vars_to_standardize A list of variable names to create standardized columns of
    @return A pandas df with the prepped data
    """
    dat = set_time_variables(dat, time_variables)
    dat = append_standardized_vars(dat, vars_to_standardize)
    return dat

def get_data(filename, datetime_colname, time_variables, vars_to_standardize):
    """
    @param dat A pandas df
    @param time_variables A list of time variables to create
    @param vars_to_standardize A list of variable names to create standardized columns of
    @return A pandas df
    """
    dat = read_in_data(filename, datetime_colname)
    dat = prepare_data(dat, time_variables, vars_to_standardize)
    return dat

def split_sample(dat_features, dat_target, test_size):
    """Splits the sample into a training set and a test set
    @param dat_features A pandas dataframe containing feature variables
    @param dat_target A pandas series containing values of the target variable
    @param test_size The proportion of observations to be put into the test set
    @return Four pandas dataframes: 1. training features, 2. test features,
      3. training target, 4. test target
    """
    return train_test_split(
        dat_features, dat_target, test_size=test_size, random_state=RANDOM_STATE)

def poisson_predictions(dat_train, dat_train_target, dat_test, outfile):
    """Obtains and writes out the predicted values of the test set using
     poisson regression
    @param dat_train A pandas dataframe of training features
    @param dat_train_target A pandas series of values of the target
    @param dat_test A pandas dataframe with test features
    @param outfile The name of the file to output the predicted values
    """
    dat_test = sm.add_constant(dat_test, prepend=False)
    dat_train = sm.add_constant(dat_train, prepend=False)
    poisson_model = sm.Poisson(
            dat_train_target, dat_train[POISSON_FEATURES]).fit(method='newton', maxiter=MAX_ITER)
    poisson_predict = poisson_model.predict(dat_test[POISSON_FEATURES])
    np.savetxt(outfile, poisson_predict, delimiter=",")

def RMSE(target_actual, target_predicted):
    """Computes the RMSE
    @param target_actual Actual target values
    @param target_predicted Predicted target values
    @return RMSE value
    """
    return sqrt(mean_squared_error(target_actual, target_predicted))

def rmse_poisson_subsamples(dat_train_subsample, dat_train_target_subsample
        , dat_test_subsample, dat_test_target_subsample):
    """Using subsamples, computes the RMSE of actual and predcited target values
     using Poisson Regression
    @param dat_train_subsample A subsample of training features
    @param dat_train_target_subsample A series of training target values
    @param dat_test_subsample A subsample of test features
    @param dat_test_target_subsample A series of test target values
    @return RMSE value
    """
    dat_train_subsample = sm.add_constant(dat_train_subsample, prepend=False)
    dat_test_subsample = sm.add_constant(dat_test_subsample, prepend=False)
    poisson_model_subsample = sm.Poisson(
            dat_train_target_subsample, dat_train_subsample[POISSON_FEATURES]).fit(
                    method='newton', maxiter=MAX_ITER)
    return RMSE(dat_test_target_subsample, poisson_model_subsample.predict(dat_test_subsample[POISSON_FEATURES]))

def score_value(model_object, dat, features, target):
    """Returns the score value for the given model and data
    @param model_object A scikit-learn model object that has been fitted
    @param dat A pandas dataframe of features
    @param features A list of features to pick from df
    @param target A pandas series of the target
    @return the score value
    """
    model_object.fit(dat[features], target)
    return model_object.score(dat[features], target)

def rmse_subsample(model_object, dat_train_subsample, dat_train_target_subsample
        , dat_test_subsample, dat_test_target_subsample, features):
    """
    @param model_object A scikit-learn model object that has been fitted
    @param dat_train_subsample A subsample of training features
    @param dat_train_target_subsample A series of training target values
    @param dat_test_subsample A subsample of test features
    @param dat_test_target_subsample A series of test target values
    @param features A list of features to use
    @return the RMSE value
    """
    model_object.fit(dat_train_subsample[features], dat_train_target_subsample)
    preds = model_object.predict(dat_test_subsample[features])
    return RMSE(dat_test_target_subsample, preds)

def writeout_predictions(model_object, dat_train, dat_train_target, dat_test, features, outfile):
    model_object.fit(dat_train[features], dat_train_target)
    preds = model_object.predict(dat_test[features])
    np.savetxt(outfile, preds, delimiter=',')


def main():
    # reading in training and test data sets
    dat = get_data(filename_train, DATETIME_COLNAME, TIME_VARIABLES, VARS_TO_STANDARDIZE)
    dat_test = get_data(filename_test, DATETIME_COLNAME, TIME_VARIABLES, VARS_TO_STANDARDIZE)
    # pulling out the target variable
    dat_train_target_df = dat[TARGET]

    # splitting the training data set into a subsample of training and test data
    features_train_subsample, features_test_subsample, target_train_subsample, target_test_subsample = split_sample(
        dat, dat_train_target_df, TEST_SAMPLE_PERCENTAGE)

    # creating model objects
    lr = linear_model.LinearRegression()
    ridge = linear_model.Ridge()
    lasso = linear_model.Lasso()
    rf = RandomForestRegressor(n_estimators=200)

    # printing RMSE values to console
    print("RMSE for poisson regression: %.2f" % rmse_poisson_subsamples(features_train_subsample, target_train_subsample
        , features_test_subsample, target_test_subsample))
    print("RMSE for linear regression: %.2f" % rmse_subsample(lr, features_train_subsample, target_train_subsample
        , features_test_subsample, target_test_subsample, LR_FEATURES))
    print("RMSE for ridge regression: %.2f" % rmse_subsample(ridge, features_train_subsample, target_train_subsample
        , features_test_subsample, target_test_subsample, RIDGE_FEATURES))
    print("RMSE for lasso regression: %.2f" % rmse_subsample(lasso, features_train_subsample, target_train_subsample
        , features_test_subsample, target_test_subsample, LASSO_FEATURES))
    print("RMSE for random forest regression: %.2f" % rmse_subsample(rf, features_train_subsample, target_train_subsample
        , features_test_subsample, target_test_subsample, RF_FEATURES))

    # outputting predicted values
    poisson_predictions(dat, dat_train_target_df, dat_test, outfile_poisson_preds)
    writeout_predictions(lr, dat, dat_train_target_df, dat_test, LR_FEATURES, outfile_lr_preds)
    writeout_predictions(ridge, dat, dat_train_target_df, dat_test, RIDGE_FEATURES, outfile_ridge_preds)
    writeout_predictions(lasso, dat, dat_train_target_df, dat_test, LASSO_FEATURES, outfile_lasso_preds)
    writeout_predictions(rf, dat, dat_train_target_df, dat_test, RF_FEATURES, outfile_rf_preds)

if __name__=="__main__":
    main()
