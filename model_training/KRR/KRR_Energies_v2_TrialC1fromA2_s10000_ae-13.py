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
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import learning_curve

try:
    import qml
    from qml.kernels import gaussian_kernel
    from qml.kernels import laplacian_kernel
    from qml.math import cho_solve
    from qml.representations import get_slatm_mbtypes
    
    print("QML version:",qml.__version__)
except ImportError:
    print("Failed to find QML")
    print("Please follow instructions here: http://www.qmlcode.org/installation.html")


##############################################################################
# Functions
##############################################################################


##############################################################################
# Kernel Settings
##############################################################################

#kernel_choice = gaussian_kernel
sigma = 10000               # standard deviation of the kernel
gamma = 1.0/(2*sigma**2)    # controlled by sigma
alpha = 1e-13                # size of the regularisation
filename = 'KRR_Energies_v2'
save_directory = 'TrialC1'
version = 'TrialC1fromA2_s10000_ae-13'
Test_Size = 0.33
Random_State = 42


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

X = np.load('./CreateFeatures_v20_fAng_fNH_B0p07_A0p07/GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_X.npy', allow_pickle=True)
y = np.load('./CreateFeatures_v20_fAng_fNH_B0p07_A0p07/GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_y.npy', allow_pickle=True)

y_round = np.round(y, decimals=5)

y_list = list(y_round)

#np.set_printoptions(threshold=sys.maxsize)
#print(X)
#sys.quit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Test_Size, random_state=Random_State)

kernel = KernelRidge(gamma=gamma, kernel='rbf', alpha=alpha)

kernel.fit(X_train, y_train)

predictions = kernel.predict(X_test)

data = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y_test, \
                           'Predicted Energy [kcal/mol]': predictions})

data.to_csv(path_or_buf='./'+save_directory+'/'+filename+'_'+version+'_'+'predicted_vs_actual_energy.csv')
pickle.dump(kernel, open('./'+save_directory+'/'+filename+'_'+version+'.plk', 'wb'))


print('')
print('Training completed')
print('MAE:', metrics.mean_absolute_error(y_test, predictions), 'kcal/mol')
print('MSE:', metrics.mean_squared_error(y_test, predictions), 'kcal/mol')
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)), 'kcal/mol')
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
