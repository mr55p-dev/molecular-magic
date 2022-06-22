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
import numpy as np
import logging
import pickle

from sklearn import metrics

from datetime import datetime

###############################################################################
# Set up Linear Regression
###############################################################################

code_name = 'LR_Energies_v3'
save_directory = 'TrialD2'
version_no = 'TrialD2_Energy_Test12'
file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Test12'
X_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Test12_v1_f761_X.npy'
y_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Test12_v1_f761_y.npy'
X_trainstd = 41.09088672608655

linear_filename = 'LR_Energies_v3_TrialD1_Reg0p00100_Ep150000.plk'
linear_directory = 'TrialD1'

logging.basicConfig(filename=save_directory+'/'+code_name+'_'+version_no+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
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

X = np.load('./'+file_directory+'/'+X_name, allow_pickle=True)
y = np.load('./'+file_directory+'/'+y_name, allow_pickle=True)

X_scaled = X/X_trainstd

print('')
print('X:')
print(X)
print('X_scaled:')
print(X_scaled)
print('y')
print(y)
logging.info('')
logging.info('X:')
logging.info(X)
logging.info('X_scaled:')
logging.info(X_scaled)
logging.info('y')
logging.info(y)

reg = pickle.load(open('./'+linear_directory+'/'+linear_filename, 'rb'))

reg_coeffs = reg.coef_
reg_interc = reg.intercept_

predictions = reg.predict(X_scaled)

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

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y, \
                           'Predicted Energy [kcal/mol]': predictions})

data.to_csv(path_or_buf='./'+save_directory+'/'+code_name+'_'+version_no+'_predvsreal.csv')

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

print('')
print('Training completed')
print('MAE:', metrics.mean_absolute_error(y, predictions), 'kcal/mol')
print('MSE:', metrics.mean_squared_error(y, predictions), 'kcal/mol')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y, predictions)), 'kcal/mol')
print('')

logging.info('')
logging.info('Training completed')
logging.info('MAE in kcal/mol')
logging.info(metrics.mean_absolute_error(y, predictions))
logging.info('MSE in kcal/mol')
logging.info(metrics.mean_squared_error(y, predictions))
logging.info('RMSE in kcal/mol')
logging.info(np.sqrt(metrics.mean_squared_error(y, predictions)))
logging.info('')