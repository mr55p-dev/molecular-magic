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


from pathlib import Path
import sys
print("Printing version info for help reporting bugs")
print("Python version:", sys.version)
import pandas as pd
import numpy as np
import logging
import pickle

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics

from datetime import datetime

###############################################################################
# Set up Linear Regression
###############################################################################

code_name = 'LR_Energies_v3'
save_directory = 'TrialD1'
version_no = 'TrialD1_Reg0p10000_Ep100000'
file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07'
X_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_X.npy'
y_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_y.npy'
epochs = 200000
Regularisation = 0.10
loss_function = 'squared_loss'

logging.basicConfig(filename='logs/lr.log', level=logging.DEBUG, format='%(message)s', filemode='w')
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")

##############################################################################
# Functions
##############################################################################


##############################################################################
# Main Code
##############################################################################

print('')      
print('**********************************************************************')
print('')
print('LR_Energies_v3.py')
print('')
print('Using numpy objects GDB_X.npy and GDB_y.npy performs Linear Regression ')
print('rmachine learning on molecule energies')
print('')
print('Author: Sanha Lee')
print('University of Cambridge')
print('')
print('**********************************************************************')
print('')
logging.info('')      
logging.info('**********************************************************************')
logging.info('')
logging.info('LR_Energies_v3.py')
logging.info('')
logging.info('Using numpy objects GDB_X.npy and GDB_y.npy performs Linear Regression ')
logging.info('rmachine learning on molecule energies')
logging.info('')
logging.info('Author: Sanha Lee')
logging.info('University of Cambridge')
logging.info('')
logging.info('**********************************************************************')
logging.info('')

# import feature and target vectors
data_basepath = Path("static_data/create_features_output/data/")
X = np.load(data_basepath / "features.npy", allow_pickle=True)
y = np.load(data_basepath / "labels.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#X_scalar = preprocessing.StandardScaler(with_mean=True).fit(X_train)
#X_trainscaled = X_scalar.transform(X_train)
#X_testscaled = X_scalar.transform(X_test)

X_trainmean = X_train.mean()
X_trainstd = X_train.std()
y_trainmean = y_train.mean()
y_trainstd = y_train.std()

print('')
print('X_train mean:')
print(X_trainmean)
print('X_train stdev:')
print(X_trainstd)
print('')
print('y_train mean:')
print(y_trainmean)
print('y_train stdev:')
print(y_trainstd)
logging.info('')
logging.info('X_train mean:')
logging.info(X_trainmean)
logging.info('X_train stdev:')
logging.info(X_trainstd)
logging.info('')
logging.info('y_train mean:')
logging.info(y_trainmean)
logging.info('y_train stdev:')
logging.info(y_trainstd)

#X_trainscaled = X_train - X_trainmean
#X_trainscaled = X_trainscaled/X_trainstd
X_trainscaled = X_train/X_trainstd

#X_testscaled = X_test - X_trainmean
#X_testscaled = X_testscaled/X_trainstd
X_testscaled = X_test/X_trainstd

print('')
print('X_train:')
print(X_train)
print('X_trainscaled:')
print(X_trainscaled)
print('X_test:')
print(X_test)
print('X_testscaled:')
print(X_testscaled)
print('y_train')
print(y_train)
print('y_test')
print(y_test)
logging.info('')
logging.info('X_train:')
logging.info(X_train)
logging.info('X_trainscaled:')
logging.info(X_trainscaled)
logging.info('X_test:')
logging.info(X_test)
logging.info('X_testscaled:')
logging.info(X_testscaled)
logging.info('y_train')
logging.info(y_train)
logging.info('y_test')
logging.info(y_test)

reg = SGDRegressor(alpha=Regularisation, loss=loss_function, learning_rate='invscaling', max_iter=epochs).fit(X_trainscaled, y_train)

reg_coeffs = reg.coef_
reg_interc = reg.intercept_

predictions = reg.predict(X_testscaled)

print('')
print('predictions:')
print(predictions)
logging.info('')
logging.info('predictions:')
logging.info(predictions)

'''
descaled_predictions = []

for item in predictions:
    descaled_predictions += [item * y_trainstd]

print('descaled_predictions:')
print(descaled_predictions)
'''

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y_test, \
                           'Predicted Energy [kcal/mol]': predictions})

output_path = Path("static_data/lr/")
output_path.mkdir(parents=True)
data.to_csv(path_or_buf=output_path / "predictions.csv")

print('')
print('The fitted weights are:')
print(reg_coeffs)
print('minimum weight:')
print(min(reg_coeffs))
print('maximum weight:')
print(max(reg_coeffs))

print('')
print('The fitted intercept is:')
print(reg_interc)

logging.info('')
logging.info('The fitted weights are:')
logging.info(reg_coeffs)
logging.info('minimum weight:')
logging.info(min(reg_coeffs))
logging.info('maximum weight:')
logging.info(max(reg_coeffs))

logging.info('')
logging.info('The fitted intercept is:')
logging.info(reg_interc)

pickle.dump(reg, open('./'+save_directory+'/'+code_name+'_'+version_no+'.plk', 'wb'))

print('')
print('Training completed')
print('MAE:', metrics.mean_absolute_error(y_test, predictions), 'kcal/mol')
print('MSE:', metrics.mean_squared_error(y_test, predictions), 'kcal/mol')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)), 'kcal/mol')
print('')

logging.info('')
logging.info('Training completed')
logging.info('MAE in kcal/mol')
logging.info(metrics.mean_absolute_error(y_test, predictions))
logging.info('MSE in kcal/mol')
logging.info(metrics.mean_squared_error(y_test, predictions))
logging.info('RMSE in kcal/mol')
logging.info(np.sqrt(metrics.mean_squared_error(y_test, predictions)))
logging.info('')