"""This is the main script that performs the execution of the basic bootstrap simulation!
"""
import numpy as np
import pandas as pd
import random
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from utils import bootstrap


#Set and initialize required global variables
#-----------------------------------------------------------------------------
N_OBS = 3000
N_SIM = 1

logreg_acc_list = []
logreg_f1_list = []
nn_acc_list = []
nn_f1_list = []

LOGISTIC_PARAMETERS = {'random_state': 2024272}
NEURAL_NETWORK_PARAMETERS = {}
TEST_SIZE = 0.3



#Set seed 
#-----------------------------------------------------------------------------
random.seed(2024272)
np.random.seed(2024272)


#Load in the data and process 
#-----------------------------------------------------------------------------
data_path = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Data', 'Customer_Churn.csv')
df = pd.read_csv(data_path)
column_names = df.columns

#Set the Y variable (Churn) to be the first colum
col_order = ['Churn'] + [col for col in column_names if col != 'Churn']
df = df[col_order]
X = df.to_numpy()


#Perform the simulation using the specified model:
#-----------------------------------------------------------------------------
for b in range(N_SIM):
    #Obtain a bootstrap sample
    x_b = bootstrap(X, N_OBS)

    #Make a train-test split
    xb_train, xb_test, yb_train, yb_test = train_test_split(x_b[:,1:], x_b[:,0], test_size= TEST_SIZE)

    #Logistic Regression:
    #Train the model using the train set:
    logreg_obj = LogisticRegression(**LOGISTIC_PARAMETERS)
    logreg_obj.fit(xb_train, yb_train)

    #Evaluate the test dataset:
    y_pred = logreg_obj.predict(xb_test)

    #Compute accuracy measures and store these results:
    logreg_f1 = f1_score(yb_test, y_pred)
    logreg_acc = accuracy_score(yb_test, y_pred)

    logreg_f1_list.append(logreg_f1)
    logreg_acc_list.append(logreg_acc)


    #Neural Network
    #Fit and train model here!




#Store the results in a seperate folder
storage_loc = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Results', 'bootstrap')

#Store logistic regression results
with open(storage_loc + '/logistic_regression_accuracy.pkl', 'wb') as f:
    pickle.dump(logreg_acc_list, f)
with open(storage_loc + '/logistic_regression_f1.pkl', 'wb') as f:
    pickle.dump(logreg_f1_list, f)
with open(storage_loc + '/neural_network_accuracy.pkl', 'wb') as f:
    pickle.dump(nn_acc_list, f)
with open(storage_loc + '/neural_network_f1.pkl', 'wb') as f:
    pickle.dump(nn_f1_list, f)



print('The execution has finished! See the folder "Results" for the simulation results!')
