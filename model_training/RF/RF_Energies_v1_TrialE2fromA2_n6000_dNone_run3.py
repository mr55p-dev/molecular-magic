#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:35:37 2019

KRR_Energies.py

Using numpy objects GDB_X.npy and GDB_y.npy performs Kernel Ridge Regression 
machine learning on molecule energies

@author: sl884
University of Cambridge

"""

##############################################################################
# Import Modules
##############################################################################


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
save_directory = 'TrialE2'
version = 'TrialE2fromA2_n6000_dNone_run3'
Test_Size = 0.33
Random_State = 42

# Hyperparameters to tune
Max_Samples = 0.999          # percentage of the training data to make boostrap sample
Max_Features = 761         # number of features that is randomly sampled
N_Estimators = 6000         # number of trees
Max_Depth = None             # maximum tree depth


logging.basicConfig(filename=save_directory+'/'+filename+'_'+version+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
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

X = np.load('./CreateFeatures_v20_fAng_fNH_B0p07_A0p07/GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_X.npy', allow_pickle=True)
y = np.load('./CreateFeatures_v20_fAng_fNH_B0p07_A0p07/GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_y.npy', allow_pickle=True)

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

data.to_csv(path_or_buf='./'+save_directory+'/'+filename+'_'+version+'_'+'predicted_vs_actual_energy.csv')
pickle.dump(RF_model, open('./'+save_directory+'/'+filename+'_'+version+'.plk', 'wb'))


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
