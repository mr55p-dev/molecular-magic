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


##############################################################################
# Functions
##############################################################################


##############################################################################
# Kernel Settings
##############################################################################

#kernel_choice = gaussian_kernel
sigma = 3000                # standard deviation of the kernel
gamma = 1.0/(2*sigma**2)    # controlled by sigma
alpha = 1e-11                # size of the regularisation
filename = 'KRR_Energies_v2'
save_directory = 'TrialG4'
version = 'TrialG4_Energy_vTestOriginalC1'
Test_Size = 0.33
Random_State = 42

kernel_filename = 'KRR_Energies_v2_TrialC1fromA2_s3000_ae-11.plk'
kernel_directory = 'TrialC1'

file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Distv2_0p01_Rep1'
Xtest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Distv2_0p01_Rep1_v1_f761_X_test.npy'
ytest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_Distv2_0p01_Rep1_v1_f761_y_test.npy'


logging.basicConfig(filename=save_directory+'/'+filename+'_'+version+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")


##############################################################################
# Main Code
##############################################################################

print('')      
print('**********************************************************************')
print('')
print('KRR_Energies.py')
print('')
print('Using numpy objects GDB_X.npy and GDB_y.npy performs Kernel Ridge Regression ')
print('rmachine learning on molecule energies')
print('')
print('Author: Sanha Lee')
print('University of Cambridge')
print('')
print('**********************************************************************')
print('')
print('Using following settings to calculate the Kernels')
print('sigma = '+str(sigma))
print('gamma = '+str(gamma))
print('alpha = '+str(alpha))
print('')



logging.info('')      
logging.info('**********************************************************************')
logging.info('')
logging.info('KRR_Energies.py')
logging.info('')
logging.info('Using numpy objects GDB_X.npy and GDB_y.npy performs Kernel Ridge Regression ')
logging.info('rmachine learning on molecule energies')
logging.info('')
logging.info('Author: Sanha Lee')
logging.info('University of Cambridge')
logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('Using following settings to calculate the Kernels')
logging.info('sigma = '+str(sigma))
logging.info('gamma = '+str(gamma))
logging.info('alpha = '+str(alpha))
logging.info('')


# import feature and target vectors

X = np.load('./'+file_directory+'/'+Xtest_name, allow_pickle=True)
y = np.load('./'+file_directory+'/'+ytest_name, allow_pickle=True)

#np.set_printoptions(threshold=sys.maxsize)
#print(X)
#sys.quit()

y_rounded = np.round(y,5)
y_load_rounded = y_rounded.tolist()

kernel = pickle.load(open('./'+kernel_directory+'/'+kernel_filename, 'rb'))

predictions = kernel.predict(X)

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y, \
                           'Predicted Energy [kcal/mol]': predictions})

data.to_csv(path_or_buf='./'+save_directory+'/'+filename+'_'+version+'_'+'predicted_vs_actual_energy.csv')


data['error'] = abs(data['Actual Energy [kcal/mol]'] - data['Predicted Energy [kcal/mol]'])
data = data.sort_values(['error'], ascending=[False])

data_largeerror = data[data.error > 0.0]

print('')
print('Largest absolute errors in kcal mol:')
print(data_largeerror)
logging.info('')
logging.info('Largest absolute errors in kcal mol:')
logging.info(data_largeerror)

# error_indexes will contain the index of the structures giving large error from the original y vector

print('')
print('looping over all predictions where the error is greater than 1 kcal mol')
logging.info('')
logging.info('looping over all predictions where the error is greater than 1 kcal mol')

error_indexes = []

for df_index, df_row in data_largeerror.iterrows():
    y_rowenergy = df_row['Actual Energy [kcal/mol]']
    error_indexes += [y_load_rounded.index(round(y_rowenergy, 5))]

error_array = np.array(error_indexes)
data_largeerror['y_index'] = error_array

data_largeerror.to_csv(path_or_buf='./'+save_directory+'/'+filename+'_'+version+'_largeerrors.csv')


print('')
print('Training completed')
print('MAE:', metrics.mean_absolute_error(y, predictions), 'kcal/mol')
print('MSE:', metrics.mean_squared_error(y, predictions), 'kcal/mol')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y,predictions)), 'kcal/mol')
print('')

logging.info('')
logging.info('Training completed')
logging.info('MAE in kcal/mol:')
logging.info(metrics.mean_absolute_error(y, predictions))
logging.info('MSE: kcal/mol:')
logging.info(metrics.mean_squared_error(y, predictions))
logging.info('RMSE: kcal/mol:')
logging.info(np.sqrt(metrics.mean_squared_error(y,predictions)))
logging.info('')
