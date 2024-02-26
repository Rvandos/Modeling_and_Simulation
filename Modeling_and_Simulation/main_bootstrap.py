"""This is the main script that performs the execution of the simulation!
"""
import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from utils import bootstrap


#Set and initialize required global variables
#-----------------------------------------------------------------------------
N_OBS = 3000
N_SIM = 1

accuracy_list = []
f1_score_list = []
MODEL = 'Logistic'
MODEL_PARAMETERS = {'random_state': 3434}
TEST_SIZE = 0.3



#Set seed 
#-----------------------------------------------------------------------------




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

    #Fit a model:
    if MODEL == 'Logistic':
        #Train the model using the train set:
        logreg_obj = LogisticRegression(**MODEL_PARAMETERS)
        logreg_obj.fit(xb_train, yb_train)

        #Evaluate the test dataset:
        y_pred = logreg_obj.predict(xb_test)

        #Compute accuracy measures and store these results:

