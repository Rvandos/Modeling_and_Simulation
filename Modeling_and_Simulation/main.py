"""This is the main script that performs the execution of the simulation!
"""
import numpy as np
import sklearn
import pandas as pd
import random
import os

from utils import bootstrap, create_logistic_population

#Set seed 
#-----------------------------------------------------------------------------




#Set and initialize required global variables
#-----------------------------------------------------------------------------
N_OBS = 2000
N_SIM = 1000




#Load in the data and process 
#-----------------------------------------------------------------------------
data_path = os.path.join(os.getcwd(),'Modeling_and_Simulation', 'Data', 'Customer_Churn.csv')
df = pd.read_csv(data_path)
column_names = df.columns

#Set the Y variable (Churn) to be the first colum
col_order = ['Churn'] + [col for col in column_names if col != 'Churn']
df = df[col_order]
X = df.to_numpy()


#Test of the simulation related functions:
#-----------------------------------------------------------------------------
#test of the bootstrap function:
x_b = bootstrap(X, N_OBS)
print(pd.DataFrame(x_b, columns = col_order))

#Test of generating the new population using a logistic regression.
parameters = {'class_weight': 'balanced', 
              'max_iter': 500}
X_logistic = create_logistic_population(X, parameters)
print(pd.DataFrame(X_logistic, columns = col_order))
