import numpy as np
import glob, os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from FMAPlots.colors import *
import matplotlib.pyplot as plt
from XMLEvoDynam.MLInputTools import *
from sklearn.model_selection import train_test_split

def define_parameters(system_dir, system, column_kept, n_samples):
    n_states = len(system)
    n_samples = n_samples
    n_samples_per_structure = int(n_samples/n_states)
    split_percent = int(n_samples_per_structure*0.8)
    n_features = len(np.load(f"{column_kept}"))
    return n_states, n_samples, n_samples_per_structure, split_percent, n_features


def define_train_test_groups(input_set, test_split_percent, n_samples, n_features):
    dataset = np.memmap(f'{input_set}', dtype='float32', mode='r', shape=(n_samples, n_features))
    X = dataset[:, :-2]
    y = dataset[:, -2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_percent, random_state=42)
    W_train = X_train[:,-1]
    X_train = X_train[:,:-1]
    W_test = X_test[:,-1]
    X_test = X_test[:,:-1]
    dataset.flush()
    return X_train, X_test, y_train, y_test, W_train, W_test


def export_ML_input(X_train, X_test, y_train, y_test, W_train, W_test, scalar_fit=False):
    np.save('y_train.npy', y_train.astype(int))
    np.save('y_test.npy', y_test.astype(int))
    np.save('weight_train.npy', W_train)
    np.save('weight_test.npy', W_test)
    
    if scalar_fit is True:
        print("Fir Standard Scalar to Train Set")
        scaler = preprocessing.StandardScaler().fit(X_train)
        
        print("Apply Transformation to Train Set")
        Xscaleset_train = scaler.transform(X_train)
        
        print("Apply Transformation to Test Set")
        Xscaleset_test = scaler.transform(X_test)
    
    
        np.save("xscaled_train.npy", Xscaleset_train)
        np.save("xscaled_test.npy", Xscaleset_test)
        print(f"FINAL SIZE:\nTrain Data:{np.shape(Xscaleset_train)}\nTest Data:{np.shape(Xscaleset_test)}\nTraining Label:{np.shape(y_train)}\nTest Labels:{np.shape(y_test)}\nWeight Train Data:{np.shape(W_train)}\nWeight Test Data:{np.shape(W_test)}")
    else: 
        np.save("xscaled_train.npy", X_train)
        np.save("xscaled_test.npy", X_test)
        print(f"FINAL SIZE:\nTrain Data:{np.shape(X_train)}\nTest Data:{np.shape(X_test)}\nTraining Label:{np.shape(y_train)}\nTest Labels:{np.shape(y_test)}\nWeight Train Data:{np.shape(W_train)}\nWeight Test Data:{np.shape(W_test)}")
    
    

GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)
column_names = MLInputTools.construct_feature_names(GROUP1, GROUP2)

SYSTEM = ["VGSDWRFLRGYHQYQ", "VGSDWRFLRGYHQYA"]
SYSTEM_DIR = "../01_prep"
column_kept = "../02_gen_inputset/keeper.npy"
input_kept = "../02_gen_inputset/input.npy"
test_group_size = 0.2

MLinfo = np.load("MLinfo.npy", allow_pickle=True).item()


n_states, n_samples, n_samples_per_structure, split_percent, n_features = define_parameters(SYSTEM_DIR, SYSTEM, column_kept, MLinfo['n_samples'])
X_train, X_test, y_train, y_test, W_train, W_test = define_train_test_groups(input_kept, test_group_size, n_samples, n_features)
export_ML_input(X_train, X_test, y_train, y_test, W_train, W_test, scalar_fit=False)