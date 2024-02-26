"""This script contains bootstrapping util functions that can be used within the simulations.
These include bootstrapping all data, using a logistic regression (Neural Network) to simulate 
dependent variables employing a linear (non-linear) DGM.
"""
import random
import numpy as np
from sklearn.linear_model import LogisticRegression


def bootstrap(X: np.ndarray, n_obs: int):
    """Function that resamples with replacement "n_obs" values from a 
    prespecified dataframe X. 
    
    Only the combinations of feature values that are present in the dataset are sampled!

    ARGS:
        X: numpy array containing the population dataset to be sampled from. Here rows
        refer to individuals and columns refer to feature values.
        n_obs: the number of individuals to return after resampling. 
    RETURNS:
        A bootstrap sample (numpy dataframe X) containing resampled individuals.

    SEED INFLUENCES:
        random (build-in python module)
    """
    #Randomly sample index list.
    LOW = 0
    HIGH = len(X)
    bootstrap_indices = random.choices(range(LOW, HIGH), k=n_obs)
    return X[bootstrap_indices]


#Note that X should have the dependent variable in the first column!
def create_logistic_population(X: np.ndarray, parameters: dict = None):
    """Function that uses a simple logistic regression to generate a dependent variable (Y), in which
    a linear relationship between the features and the dependent variable is imposed. 
    
    ARGS:
        X: the numpy array containing the dataset with both features and the dependent variable.
        parameters: dictionary containing the parameter values for the LogisticRegression function. Use same syntax!

    RETURNS:
        The numpy array X where the dependent variable is replaced with the newly generated values.
    
    SEED INFLUENCES:
        Sklearn has randomness within training the logistic regression, make sure to set the seed!
    """
    #Create a logistic regression object and train the model using the full dataset.
    y_old = X[:,0]
    x_features = X[:,1:]

    if parameters != None:
        logreg_obj = LogisticRegression(**parameters)
    else:
        print('You should set a seed for SKLEARN! Pass parameters as input for LogisticRegression()!')
        logreg_obj = LogisticRegression()
    
    logreg_obj.fit(x_features, y_old)

    #Then fit the features to get new dependent variables.
    y_new = logreg_obj.predict(x_features)

    #replace the dependent variable with the new values (assumed to be the first column of X)
    X_new_population = np.hstack((y_new.reshape(-1,1), x_features))

    #Print information on the new population:
    print(f'The number of people that churn in the original dataset {y_old.sum()} & new dataset {y_new.sum()}')
    print(f'The number of different values due to generation is equal to {np.abs(y_new - y_old).sum()}')
    return X_new_population

