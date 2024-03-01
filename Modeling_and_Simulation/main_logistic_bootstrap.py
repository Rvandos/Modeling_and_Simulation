"""This is the main script that performs the execution of the logistic bootstrap simulation. A new population 
is created, where a linear relationship is imposed between the set of features and the dependent variable!
"""
import numpy as np
import pandas as pd
import random
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

from utils import bootstrap, create_logistic_population


#Set and initialize required global variables
#-----------------------------------------------------------------------------
N_OBS = 3000
N_SIM = 1

logreg_acc_list = []
logreg_f1_list = []
nn_acc_list = []
nn_f1_list = []

REMOVE_FEATURES = ['Age Group']
POPULATION_LOGISTIC_PARAMETERS = {}
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

#Create a new population using a logistic regression:
X_new = create_logistic_population(X, POPULATION_LOGISTIC_PARAMETERS)


#Perform the simulation on the newly generated population:
for b in range(N_SIM):
    #Obtain a bootstrap sample
    xb_new = bootstrap(X_new, N_OBS)

    #Make a train-test split
    xb_train, xb_test, yb_train, yb_test = train_test_split(xb_new[:,1:], xb_new[:,0], test_size= TEST_SIZE)


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
storage_loc = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Results', 'logistic_bootstrap')

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

