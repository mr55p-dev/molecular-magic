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
save_directory = 'TrialC4'
version = 'TrialC4_G_Test12'
Test_Size = 0.33
Random_State = 42

kernel_filename = 'KRR_Energies_v2_TrialC3_s3000_ae-11.plk'
kernel_directory = 'TrialC3'

file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12'
Xtest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12_v1_f781_X.npy'
ytest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12_v1_f781_y.npy'


logging.basicConfig(filename='logs/krrc4.log', level=logging.DEBUG, format='%(message)s', filemode='w')
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
data_basepath = Path("static_data/create_features_output/data")
X = np.load(data_basepath / "features.npy", allow_pickle=True)
y = np.load(data_basepath / "labels.npy", allow_pickle=True)

#np.set_printoptions(threshold=sys.maxsize)
#print(X)
#sys.quit()

kernel = pickle.load(open('./'+kernel_directory+'/'+kernel_filename, 'rb'))

predictions = kernel.predict(X)

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y, \
                           'Predicted Energy [kcal/mol]': predictions})

data.to_csv(path_or_buf='./'+save_directory+'/'+filename+'_'+version+'_'+'predicted_vs_actual_energy.csv')


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