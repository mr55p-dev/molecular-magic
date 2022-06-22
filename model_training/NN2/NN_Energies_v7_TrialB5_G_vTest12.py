#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:35:37 2019

NN_Energies.py

Python code to perform neural network energy prediction on organic molecules.
The features vector and the target vector are created from CreateFeatures.py.

@author: sl884
University of Cambridge

"""


##############################################################################
# Import Modules
##############################################################################

import numpy as np
import logging
import time
start_time = time.time()
import pandas as pd
import sys

from datetime import datetime


from sklearn import metrics
from keras.models import load_model


###############################################################################
# Set up Neural Networks
###############################################################################

code_name = 'NN_Energies'
save_directory = 'TrialB5'
version_no = 'TrialB5_G_vTest12'
file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12'
model_directory = 'TrialB3'
model_name = 'NN_Energies_TrialB3_G298_Reg0p15_Adam.h5'
Xtest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12_v1_f781_X.npy'
ytest_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_G298_Test12_v1_f781_y.npy'

batch_size = 64
X_trainstd = 40.42285839734875

logging.basicConfig(filename=save_directory+'/'+code_name+'_'+version_no+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")



##############################################################################
# Functions
##############################################################################

def rectified(value):
    return max(0.0, value)


##############################################################################
# Main Code
##############################################################################

print('')      
print('**********************************************************************')
print('')
print('NN_Energies.py')
print('')
print('Using numpy objects GDB_X.npy and GDB_y.npy performs Neural Networks ')
print('machine learning on molecule energies')
print('')
print('Author: Sanha Lee')
print('University of Cambridge')
print('')
print('**********************************************************************')
print('')


logging.info('')      
logging.info('**********************************************************************')
logging.info('')
logging.info('NN_Energies.py')
logging.info('')
logging.info('Using numpy objects GDB_X.npy and GDB_y.npy performs Neural Networks ')
logging.info('machine learning on molecule energies')
logging.info('')
logging.info('Author: Sanha Lee')
logging.info('University of Cambridge')
logging.info('')
logging.info('Run date: '+formatted_datetime+'\n')
logging.info('')
logging.info('**********************************************************************')
logging.info('')
logging.info('')
logging.info('Logging NN features:')
logging.info('Xtest_name: '+str(Xtest_name))
logging.info('ytest_name: '+str(ytest_name))
logging.info('X_trainstd: '+str(X_trainstd))
logging.info('model_name: '+str(model_name))
logging.info('code_name: '+str(code_name))
logging.info('save_directory: '+str(save_directory))
logging.info('version_no: '+str(version_no))
logging.info('batch_size: '+str(batch_size))


# -- Import feature and target vectors

X_test = np.load('./'+file_directory+'/'+Xtest_name, allow_pickle=True)
y_test = np.load('./'+file_directory+'/'+ytest_name, allow_pickle=True)

y_rounded = np.round(y_test,5)
y_load_rounded = y_rounded.tolist()

X_testscaled = X_test/X_trainstd

y_testmatrix = []

for element in y_test:
    y_testmatrix.append([element])

y_testmatrix = np.array(y_testmatrix)

model = load_model('./'+model_directory+'/'+model_name)
model.summary()

predictions = model.predict(X_testscaled, batch_size=batch_size)

unwrapped_predictions = []

for pred_list in predictions:
    unwrapped_predictions += [pred_list[0]]

error_dataframe = pd.DataFrame(data= {'Actual Energy [kcal/mol]': y_test, \
                                      'Predicted Energy [kcal/mol]': unwrapped_predictions})

error_dataframe['error'] = abs(error_dataframe['Actual Energy [kcal/mol]'] - error_dataframe['Predicted Energy [kcal/mol]'])
error_dataframe = error_dataframe.sort_values(['error'], ascending=[False])

data_largeerror = error_dataframe[error_dataframe.error > 0.0]

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

data_largeerror.to_csv(path_or_buf='./'+save_directory+'/'+code_name+'_'+version_no+'_largeerrors.csv')


# -- save keras model
model.save('./'+save_directory+'/'+code_name+'_'+version_no+'.h5')

print('')
print('y_testunscaled:')
print(y_test)
print('')
print('predictions:')
print(predictions)
print('')
print('Training completed')
print('MAE: '+ str(metrics.mean_absolute_error(y_test, predictions))+' kcal/mol')
print('MSE: '+ str(metrics.mean_squared_error(y_test, predictions))+' kcal/mol')
print('RMSE: '+ str(np.sqrt(metrics.mean_squared_error(y_test, predictions)))+' kcal/mol')
print('')
print("--- %s seconds ---" % (time.time() - start_time))
print('')

logging.info('')
logging.info('y_testunscaled:')
logging.info(y_test)
logging.info('')
logging.info('predictions:')
logging.info(predictions)
logging.info('')
logging.info('Training completed')
logging.info('MAE: '+ str(metrics.mean_absolute_error(y_test, predictions))+' kcal/mol')
logging.info('MSE: '+ str(metrics.mean_squared_error(y_test, predictions))+' kcal/mol')
logging.info('RMSE: '+ str(np.sqrt(metrics.mean_squared_error(y_test, predictions)))+' kcal/mol')
logging.info('')
logging.info("--- %s seconds ---" % (time.time() - start_time))
logging.info('')
