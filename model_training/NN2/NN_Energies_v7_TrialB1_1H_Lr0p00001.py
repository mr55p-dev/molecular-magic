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
import csv
start_time = time.time()
import pandas as pd
import sys

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.optimizers import Adam
from keras.layers import Dropout


###############################################################################
# Set up Neural Networks
###############################################################################

EnergyDistLoc = 'Mol_Dist_v1'
EnerguDostFile = 'Energy_Dataframe.csv'
AtomOrder = ['C', 'H', 'N', 'O']

code_name = 'NN_Energies'
save_directory = 'TrialB1'
version_no = 'TrialB1fromA2_1H_Lr0p00001'
file_directory = 'CreateFeatures_v20_fAng_fNH_B0p07_A0p07'
X_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_X.npy'
y_name = 'GDBA_CreateFeatures_v20_fAng_fNH_B0p07_A0p07_v1_f761_y.npy'
learning_rate = 0.00001
epochs = 7000
decay_rate = 0.00003
batch_size = 64
Random_State = 50


'''

IMPORTANT

Check whether the X vector atomtypes are located at X[-15:11]
Check whether the molecular feature is in order of C N O H


'''

activation_function = 'linear'
activation_function2 = 'relu'
loss_function = 'mean_squared_error'
optimizer_type = Adam(learning_rate=learning_rate) # MAKE SURE TO EDIT THE LOG BELOW!!!

kernel_initialiser = initializers.RandomUniform(minval=-500, maxval=100)
kernel_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.1)

bias_initialiser = initializers.RandomUniform(minval=0, maxval=10)
bias_initialiser2 = initializers.RandomUniform(minval=0, maxval=0.01)

logging.basicConfig(filename=save_directory+'/'+code_name+'_'+version_no+'.log', level=logging.DEBUG, format='%(message)s', filemode='w')
datetime_now = datetime.now()
formatted_datetime = datetime_now.strftime("%Y %b %d %H:%M:%S")



##############################################################################
# Functions
##############################################################################

def rectified(value):
    return max(0.0, value)

def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate



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
logging.info('EnergyDistLoc: '+str(EnergyDistLoc))
logging.info('EnerguDostFile: '+str(EnerguDostFile))
logging.info('AtomOrder: '+str(AtomOrder))
logging.info('code_name: '+str(code_name))
logging.info('save_directory: '+str(save_directory))
logging.info('version_no: '+str(version_no))
logging.info('file_directory: '+str(file_directory))
logging.info('X_name: '+str(X_name))
logging.info('y_name: '+str(y_name))
logging.info('learning_rate: '+str(learning_rate))
logging.info('epochs: '+str(epochs))
logging.info('decay_rate: '+str(decay_rate))
logging.info('batch_size: '+str(batch_size))
logging.info('Random_State: '+str(Random_State))
logging.info('activation_function: '+str(activation_function))
logging.info('activation_function2: '+str(activation_function2))
logging.info('loss_function: '+str(loss_function))
logging.info('optimizer_type: '+str(optimizer_type))
logging.info('kernel_initialiser: '+str(kernel_initialiser))
logging.info('kernel_initialiser2: '+str(kernel_initialiser2))
logging.info('bias_initialiser: '+str(bias_initialiser))
logging.info('bias_initialiser2: '+str(bias_initialiser2))

# -- Import feature and target vectors

X_load = np.load('./'+file_directory+'/'+X_name, allow_pickle=True)
y_load = np.load('./'+file_directory+'/'+y_name, allow_pickle=True)

y_rounded = np.round(y_load,5)

X_load = X_load.tolist()
y_load = y_load.tolist()
y_load_rounded = y_rounded.tolist()

# -- Import Molecular Formulas

Edist_df = pd.read_csv('Mol_Dist_v1/Energy_Dataframe.csv')

Mol_Forms = Edist_df['Mol_Forms']
Sample_N = Edist_df['Number_of_Samples']


# -- WARNING: set randomises the order
Unique_Mol_Forms = list(set(Mol_Forms))

print('')
print('Found '+str(len(Unique_Mol_Forms))+' unique molecular formulas')
logging.info('')
logging.info('Found '+str(len(Unique_Mol_Forms))+' unique molecular formulas')

# -- Divide the data into sub categories

X_dict = {}
y_dict = {}

for MolForm in Unique_Mol_Forms:
    X_dict[MolForm] = []
    y_dict[MolForm] = []

first10_AtomList100 = []


for index in range(len(y_load)):
    
    X_vector = X_load[index]
    y_value = y_load[index]
    
    AtomList100 = X_vector[-15:-11]
    #print('AtomList')
    #print(AtomList100)
    
    if index < 10:
        first10_AtomList100 += [AtomList100]
    
    AtomList = [int(x_i/100) for x_i in AtomList100]
    
    Molecular_Formula = 'C'+str(AtomList[AtomOrder.index('C')])+'N'+str(AtomList[AtomOrder.index('N')])+'O'+str(AtomList[AtomOrder.index('O')])+'H'+str(AtomList[AtomOrder.index('H')])
    
    X_dict[Molecular_Formula].append(X_vector)
    y_dict[Molecular_Formula].append(y_value)


print('')
print('Check whether AtomList100 is valid for first 10 molecules:')
print(first10_AtomList100)
print('')

logging.info('')
logging.info('Check whether AtomList100 is valid for first 10 molecules:')
logging.info(first10_AtomList100)
logging.info('')

#print(X_dict)/
#print(y_dict)


X_train = []
X_test = []
y_train = []
y_test = []

for MolForm in Unique_Mol_Forms:
    
    X_MolFormList = X_dict[MolForm]
    y_MolFormList = y_dict[MolForm]
    print(MolForm)
    logging.info(MolForm)
    #print('X_MolFormList: ',np.array(X_MolFormList))
    #print('y_MolFormList: ',np.array(y_MolFormList))
    print('Number of molecules for this molecular formula: ', len(y_MolFormList))
    logging.info('Number of molecules for this molecular formula: '+str(len(y_MolFormList)))
    
    if len(y_MolFormList) == 1:
        print('Only one molecule, not included in Train or Test set')
        logging.info('Only one molecule, not included in Train or Test set')
        pass
    elif len(y_MolFormList) == 2:
        X_MolFormTrain, X_MolFormTest, y_MolFormTrain, y_MolFormTest = train_test_split(X_MolFormList, y_MolFormList, test_size=0.5, random_state=Random_State)
        #print('Only two molecules')
        #print('X_MolFromTrain: ',np.array(X_MolFormTrain))
        #print('X_MolFromTest: ',np.array(X_MolFormTest))
        #print('y_MolFromTrain: ',np.array(y_MolFormTrain))
        #print('y_MolFromTest: ',np.array(y_MolFormTest))
        print('y_MolFromTrain: '+str(len(y_MolFormTrain))+' molecules')
        print('y_MolFromTest: '+str(len(y_MolFormTest))+' molecules')
        logging.info('y_MolFromTrain: '+str(len(y_MolFormTrain))+' molecules')
        logging.info('y_MolFromTest: '+str(len(y_MolFormTest))+' molecules')
        X_train += X_MolFormTrain
        X_test += X_MolFormTest
        y_train += y_MolFormTrain
        y_test += y_MolFormTest
    else:
        X_MolFormTrain, X_MolFormTest, y_MolFormTrain, y_MolFormTest = train_test_split(X_MolFormList, y_MolFormList, test_size=0.33, random_state=Random_State)
        #print('More than two molecules')
        #print('X_MolFromTrain: ',np.array(X_MolFormTrain))
        #print('X_MolFromTest: ',np.array(X_MolFormTest))
        #print('y_MolFromTrain: ',np.array(y_MolFormTrain))
        #print('y_MolFromTest: ',np.array(y_MolFormTest))
        print('y_MolFromTrain: '+str(len(y_MolFormTrain))+' molecules')
        print('y_MolFromTest: '+str(len(y_MolFormTest))+' molecules')
        logging.info('y_MolFromTrain: '+str(len(y_MolFormTrain))+' molecules')
        logging.info('y_MolFromTest: '+str(len(y_MolFormTest))+' molecules')
        X_train += X_MolFormTrain
        X_test += X_MolFormTest
        y_train += y_MolFormTrain
        y_test += y_MolFormTest

    


#sys.exit()
# -- Sanity Check:

print('')
print('Checking whether X_vector and y_energy are correctly assigned')
logging.info('')
logging.info('Checking whether X_vector and y_energy are correctly assigned')
removed_energies = []

for index1 in range(len(y_load)):
    
    #print(index1)
    
    X_vector = X_load[index1]
    y_energy = y_load[index1]
    #print(y_energy)
    
    if y_energy in y_train:
        #print('This y_energy is in the Training set')
        index2 = y_train.index(y_energy)
        X_vector_test = X_train[index2]
        
        if X_vector != X_vector_test:
            print('ERROR: X_vector does not match X_vector_test')
            print('X_vector: ', X_vector)
            print('X_vector_test: ', X_vector)
            logging.info('ERROR: X_vector does not match X_vector_test')
            logging.info('X_vector: ')
            logging.info(X_vector)
            logging.info('X_vector_test: ')
            logging.info(X_vector)
            sys.exit()
        
    elif y_energy in y_test:
        #print('This y_energy is in the Test set')
        index2 = y_test.index(y_energy)
        X_vector_test = X_test[index2]
        
        if X_vector != X_vector_test:
            print('ERROR: X_vector does not match X_vector_test')
            print('X_vector: ', X_vector)
            print('X_vector_test: ', X_vector)
            logging.info('ERROR: X_vector does not match X_vector_test')
            logging.info('X_vector: ')
            logging.info(X_vector)
            logging.info('X_vector_test: ')
            logging.info(X_vector)
            sys.exit()
        
    else:
        removed_energies += [y_energy]
        #print('This y_energy has been removed')

    #sys.stdout.write('\r')
    #sys.stdout.write(str(round(100.0*float(index1)/float(len(y_load)), 1))+'%')
    #sys.stdout.flush()

print('')
print('Passed assignement check')
logging.info('')
logging.info('Passed assignement check')

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print('')
print('Matrix X_train preview:')    
print(X_train)
print('')
print('Matrix X_test preview:')    
print(X_test)

print('')
print('Matrix y_train preview:')    
print(y_train)
print('')
print('Matrix y_test preview:')    
print(y_test)

logging.info('')
logging.info('Matrix X_train preview:')    
logging.info(X_train)
logging.info('')
logging.info('Matrix X_test preview:')    
logging.info(X_test)

logging.info('')
logging.info('Matrix y_train preview:')    
logging.info(y_train)
logging.info('')
logging.info('Matrix y_test preview:')    
logging.info(y_test)

#sys.exit()

print('')
print('Training and Test set split completed')
print('Number of molecules removed because the molecular formula contains only one molecule: ',len(removed_energies))
print('Total number of molecules available for NN model: ', (len(y_train)+len(y_test)))
print('Number of molecules in the training set: ', len(y_train))
print('Number of molecules in the test set: ', len(y_test))
print('Test set percentage: ', round(len(y_test)/(len(y_train)+len(y_test))*100,2) )

logging.info('')
logging.info('Training and Test set split completed')
logging.info('Number of molecules removed because the molecular formula contains only one molecule: '+str(len(removed_energies)))
logging.info('Total number of molecules available for NN model: '+str((len(y_train)+len(y_test))))
logging.info('Number of molecules in the training set: '+str(len(y_train)))
logging.info('Number of molecules in the test set: '+str(len(y_test)))
logging.info('Test set percentage: '+str(round(len(y_test)/(len(y_train)+len(y_test))*100,2) ))

print('')
print('Saving training and test set X and y matrices')
logging.info('')
logging.info('Saving training and test set X and y matrices')

np.save('./'+save_directory+'/'+code_name+'_'+version_no+'_X_train', X_train, allow_pickle=True)
np.save('./'+save_directory+'/'+code_name+'_'+version_no+'_X_test', X_test, allow_pickle=True)
np.save('./'+save_directory+'/'+code_name+'_'+version_no+'_y_train', y_train, allow_pickle=True)
np.save('./'+save_directory+'/'+code_name+'_'+version_no+'_y_test', y_test, allow_pickle=True)



X_trainmean = X_train.mean()
X_trainstd = X_train.std()
y_trainmean = y_train.mean()
y_trainstd = y_train.std()

X_trainscaled = X_train/X_trainstd
X_testscaled = X_test/X_trainstd

y_testmatrix = []
y_trainmatrix = []

for element in y_train:
    y_trainmatrix.append([element])

for element in y_test:
    y_testmatrix.append([element])
    
y_trainmatrix = np.array(y_trainmatrix)
y_testmatrix = np.array(y_testmatrix)



###############################################################################
# Construct the NN architecture
###############################################################################

no_features = len(X_trainscaled[0])
print('')
print('Number of features detected: ',str(no_features))

logging.info('')
logging.info('Number of features detected: '+str(no_features))

# model set up: build architecture
model = Sequential(name=code_name+'_'+version_no)

model.add(Dense(761, \
                input_dim=no_features, \
                activation=activation_function2, \
                kernel_initializer=kernel_initialiser2, \
                bias_initializer=bias_initialiser2, \
                kernel_regularizer=l2(0.1), \
                bias_regularizer=l2(0.1), \
                name='layer1'))

#model.add(Dropout(0.2))

#model.add(Dense(761, \
#                activation=activation_function2, \
#                kernel_initializer=kernel_initialiser, \
#                kernel_regularizer=l2(0.1), \
#                bias_regularizer=l2(0.1), \
#                bias_initializer=bias_initialiser, \
#                name='layer2'))

#model.add(Dropout(0.2))

model.add(Dense(1, \
                activation=activation_function, \
                kernel_initializer=kernel_initialiser, \
                bias_initializer=bias_initialiser, \
                kernel_regularizer=l2(0.1), \
                bias_regularizer=l2(0.1), \
                name='layer2'))

# model set up: loss functions, optimiser
print('')
print('Settings for neural network training')
print(model.summary())

modelsummarystringlist = []
model.summary(print_fn=lambda x: modelsummarystringlist.append(x))

logging.info('')
logging.info('Settings for neural network training')

for modelsummaryline in modelsummarystringlist:
    logging.info(modelsummaryline)

model.compile(loss=loss_function, optimizer=optimizer_type, metrics=['mean_squared_error','mean_absolute_error'])


###############################################################################
# model train
###############################################################################

model_history = model.fit(X_trainscaled, y_trainmatrix, validation_data=(X_testscaled, y_testmatrix), epochs=epochs, batch_size=batch_size, verbose=2)


with open(save_directory+'/'+code_name+'_'+version_no+'_errordata.csv','w') as errorfile:
    error_write = csv.writer(errorfile)
    epoch_data = range(0,len(model_history.history['loss']))
    csv_rows = zip(epoch_data,model_history.history['loss'],model_history.history['val_loss'],model_history.history['mean_squared_error'],model_history.history['val_mean_squared_error'])
    error_write.writerows(csv_rows)


# Evaluate Keras Model
train_error = model.evaluate(X_trainscaled, y_trainmatrix, verbose=0)
test_error = model.evaluate(X_testscaled, y_testmatrix, verbose=0)

print('train_mse: ',train_error)
print('test_mse: ',test_error)

logging.info('train_mse: ')
logging.info(str(train_error))
logging.info('test_mse: ')
logging.info(str(test_error))

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
print(y_testmatrix)
print('')
print('predictions:')
print(predictions)
print('')
print('Training completed')
print('MAE: '+ str(metrics.mean_absolute_error(y_testmatrix, predictions))+' kcal/mol')
print('MSE: '+ str(metrics.mean_squared_error(y_testmatrix, predictions))+' kcal/mol')
print('RMSE: '+ str(np.sqrt(metrics.mean_squared_error(y_testmatrix, predictions)))+' kcal/mol')
print('')
print("--- %s seconds ---" % (time.time() - start_time))
print('')

logging.info('')
logging.info('y_testunscaled:')
logging.info(y_testmatrix)
logging.info('')
logging.info('predictions:')
logging.info(predictions)
logging.info('')
logging.info('Training completed')
logging.info('MAE: '+ str(metrics.mean_absolute_error(y_testmatrix, predictions))+' kcal/mol')
logging.info('MSE: '+ str(metrics.mean_squared_error(y_testmatrix, predictions))+' kcal/mol')
logging.info('RMSE: '+ str(np.sqrt(metrics.mean_squared_error(y_testmatrix, predictions)))+' kcal/mol')
logging.info('')
logging.info("--- %s seconds ---" % (time.time() - start_time))
logging.info('')
