"""This script contains bootstrapping util functions that can be used within the simulations.
These include bootstrapping all data, using a logistic regression (Neural Network) to simulate 
dependent variables employing a linear (non-linear) DGM.
"""
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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


def create_nn_population(X: np.ndarray, n_features: int, validation_split: float):
    """Function that uses a neural network to generate a dependent variable (Y), in which
    a nonlinear relationship between the features and the dependent variable is imposed by ReLu activation functions. 
    
    ARGS:
        X: the numpy array containing the dataset with both features and the dependent variable.
        n_features: the number of features that are present in the dataset
        validation_split: the split used of the train dataset that is used for validation.

    RETURNS:
        The numpy array X where the dependent variable is replaced with the newly generated values.
    
    SEED INFLUENCES:
        Tensorflow has randomness with the network initialization, make sure to set the seed!
    """
    #Take the features
    y_old = X[:,0]
    x_features = X[:,1:]

    #Specify the Neural Network
    model = tf.model = Sequential()
    model.add(Dense(64, activation='relu', input_dim = n_features))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #Train the network
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(x_features, y_old, batch_size = 32, epochs = 100, validation_split = validation_split, callbacks = [early_stopping], verbose = 0)
    
    #Then fit the features to get new dependent variables.
    y_new = np.where(model.predict(x_features, verbose = 0) > 0.5, 1, 0)

    #replace the dependent variable with the new values (assumed to be the first column of X)
    X_new_population = np.hstack((y_new, x_features))

    #Print information on the new population:
    print(f'The number of people that churn in the original dataset {y_old.sum()} & new dataset {y_new.sum()}')
    print(f'The number of different values due to generation is equal to {np.abs(y_new - y_old.reshape(-1,1)).sum()}')
    return X_new_population