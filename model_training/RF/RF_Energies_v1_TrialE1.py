#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:35:37 2019

KRR_Energies.py

Using numpy objects GDB_X.npy and GDB_y.npy performs Kernel Ridge Regression 
machine learning on molecule energies

@author: sl884
University of Cambridge

This version sets the max number of features sampled as 500
"""

##############################################################################
# Import Modules
##############################################################################


from pathlib import Path
import sys
print("Printing version info for help reporting bugs")
print("Python version:", sys.version)

import pandas as pd
import logging
import numpy as np
import pickle

from datetime import datetime
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


##############################################################################
# Functions
##############################################################################


##############################################################################
# Kernel Settings
##############################################################################


filename = 'RF_Energies_v1'
save_directory = 'TrialE1'
version = 'TrialE1fromA2_ms0p4_mf500_run1'
Test_Size = 0.33
Random_State = 42

# Hyperparameters to tune
Max_Samples = 0.4          # percentage of the training data to make boostrap sample
Max_Features = 500         # number of features that is randomly sampled
N_Estimators = 1000         # number of trees
Max_Depth = 20             # maximum tree depth


logging.basicConfig(filename='logs/rf.log', level=logging.DEBUG, format='%(message)s', filemode='w')
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")


##############################################################################
# Main Code
##############################################################################

print('')      
print('**********************************************************************')
print('')
print('RF_Energies.py')
print('')
print('Using numpy objects GDB_X.npy and GDB_y.npy performs Kernel Ridge Regression ')
print('rmachine learning on molecule energies')
print('')
print('Author: Sanha Lee')
print('University of Cambridge')
print('')
print('**********************************************************************')
print('')
print('Max_Samples:')
print(Max_Samples)
print('Max_Features:')
print(Max_Features)
print('N_Estimators:')
print(N_Estimators)
print('Max_Depth:')
print(Max_Depth)




logging.info('')      
logging.info('**********************************************************************')
logging.info('')
logging.info('RF_Energies.py')
logging.info('')
logging.info('Using numpy objects GDB_X.npy and GDB_y.npy performs Random Forest Regression ')
logging.info('rmachine learning on molecule energies')
logging.info('')
logging.info('Author: Sanha Lee')
logging.info('University of Cambridge')
logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('Max_Samples:')
logging.info(Max_Samples)
logging.info('Max_Features:')
logging.info(Max_Features)
logging.info('N_Estimators:')
logging.info(N_Estimators)
logging.info('Max_Depth:')
logging.info(Max_Depth)

# import feature and target vectors
data_basepath = Path("static_data/create_features_output/data")
X = np.load(data_basepath / "features.npy", allow_pickle=True)
y = np.load(data_basepath / "labels.npy", allow_pickle=True)

y_round = np.round(y, decimals=5)

y_list = list(y_round)

#np.set_printoptions(threshold=sys.maxsize)
#print(X)
#sys.quit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_Size, random_state=Random_State)

RF_model = RandomForestRegressor(max_samples=Max_Samples, max_features=Max_Features, n_estimators=N_Estimators, max_depth=Max_Depth)

RF_model.fit(X_train, y_train)

predictions = RF_model.predict(X_test)

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y_test, \
                           'Predicted Energy [kcal/mol]': predictions})

output_path = Path("static_data/rf/")
output_path.mkdir(parents=True, exist_ok=True)
data.to_csv(path_or_buf=output_path / "E1_predictions.csv")

with open(output_path / "E1_model.pkl", 'wb') as outfile:
    pickle.dump(RF_model, outfile)

print('')
print('Training completed')
print('MAE:', metrics.mean_absolute_error(y_test, predictions), 'kcal/mol')
print('MSE:', metrics.mean_squared_error(y_test, predictions), 'kcal/mol')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)), 'kcal/mol')
print('')

logging.info('')
logging.info('Training completed')
logging.info('MAE in kcal/mol:')
logging.info(metrics.mean_absolute_error(y_test, predictions))
logging.info('MSE: kcal/mol:')
logging.info(metrics.mean_squared_error(y_test, predictions))
logging.info('RMSE: kcal/mol:')
logging.info(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
logging.info('')