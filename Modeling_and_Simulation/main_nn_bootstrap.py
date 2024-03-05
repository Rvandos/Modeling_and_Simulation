"""This is the main script that performs the execution of the Neural Network bootstrap simulation. A new population 
is created, where a nonlinear relationship is imposed between the set of features and the dependent variable!
"""
#Surpress deprecation warnings!
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import bootstrap, create_nn_population

tf.get_logger().setLevel('ERROR')
#Set and initialize required global variables
#-----------------------------------------------------------------------------
N_OBS = 3000
N_SIM = 1000

logreg_acc_list = []
logreg_f1_list = []
nn_acc_list = []
nn_f1_list = []

REMOVE_FEATURES = ['Age Group']
POPULATION_NN_VALIDATION_SPLIT = 0.1
LOGISTIC_PARAMETERS = {'random_state': 2024272}
TEST_SIZE = 0.3
NN_VALIDATION_SPLIT = 0.1



#Set seed 
#-----------------------------------------------------------------------------
random.seed(2024272)
np.random.seed(2024272)
tf.keras.utils.set_random_seed(2024272)


#Load in the data and process 
#-----------------------------------------------------------------------------
data_path = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Data', 'Customer_Churn.csv')
df = pd.read_csv(data_path)

#Remove unwanted features:
if len(REMOVE_FEATURES) != 0:
    df = df.drop(REMOVE_FEATURES, axis=1)

#Make the status variable binary:
df['Status'] = df['Status'].replace(2,0)

#Set the Y variable (Churn) to be the first colum
column_names = df.columns
col_order = ['Churn'] + [col for col in column_names if col != 'Churn']
df = df[col_order]
X = df.to_numpy()

n_features = len(X[0])-1

#Create a new population using a logistic regression:
X_new = create_nn_population(X, n_features, validation_split= POPULATION_NN_VALIDATION_SPLIT)



#Perform the simulation on the newly generated population:
for b in tqdm(range(N_SIM)):
    #Obtain a bootstrap sample
    xb_new = bootstrap(X_new, N_OBS)

    #Make a train-test split
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb_new[:,1:], xb_new[:,0], test_size= TEST_SIZE)

    #Standardize all features (also binary, w.l.o.g. as we only care about performance).
    scaler = StandardScaler()
    xb_train_std = scaler.fit_transform(xb_train)
    xb_test_std = scaler.transform(xb_test)

    #------------------------------------------------------
    #Logistic Regression:
    #Train the model using the train set:
    logreg_obj = LogisticRegression(**LOGISTIC_PARAMETERS)
    logreg_obj.fit(xb_train_std, yb_train)

    #Evaluate the test dataset:
    y_pred = logreg_obj.predict(xb_test_std)

    #Compute accuracy measures and store these results:
    logreg_f1 = f1_score(yb_test, y_pred)
    logreg_acc = accuracy_score(yb_test, y_pred)

    logreg_f1_list.append(logreg_f1)
    logreg_acc_list.append(logreg_acc)



    #------------------------------------------------------
    #Neural Network
    #Model specification
    model = tf.model = Sequential()
    model.add(Dense(64, activation='relu', input_dim = n_features))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Train the network
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(xb_train_std, yb_train, batch_size = 32, epochs = 100, validation_split = NN_VALIDATION_SPLIT, callbacks = [early_stopping], verbose = 0)

    #Evaluate the test dataset:
    y_pred_nn = np.where(model.predict(xb_test_std, verbose =0) > 0.5, 1, 0)

    #Compute accuracy measures and store these results:
    nn_f1 = f1_score(yb_test, y_pred_nn)
    nn_acc = accuracy_score(yb_test, y_pred_nn)
    nn_acc_list.append(nn_acc)
    nn_f1_list.append(nn_f1)


#Store the results in a seperate folder
storage_loc = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Results', 'nn_bootstrap')

#Store logistic regression results
with open(storage_loc + '/logistic_regression_accuracy.pkl', 'wb') as f:
    pickle.dump(logreg_acc_list, f)
with open(storage_loc + '/logistic_regression_f1.pkl', 'wb') as f:
    pickle.dump(logreg_f1_list, f)
with open(storage_loc + '/neural_network_accuracy.pkl', 'wb') as f:
    pickle.dump(nn_acc_list, f)
with open(storage_loc + '/neural_network_f1.pkl', 'wb') as f:
    pickle.dump(nn_f1_list, f)

#Store the newly generated population
with open(storage_loc + '/new_population.csv', 'w') as f:
    pd.DataFrame(X_new, columns = col_order).to_csv(f, index = False)

print('The execution has finished! See the folder "Results" for the simulation results!')

