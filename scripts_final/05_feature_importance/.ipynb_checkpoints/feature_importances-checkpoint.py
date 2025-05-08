from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from FMAPlots.colors import *
import random
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--depth", type=int, help="increase output verbosity")
parser.add_argument("-e", "--estimators", type=int, help="increase output verbosity")
args = parser.parse_args()

ncv = 16
MAX_DEPTH = args.depth
N_ESTIMATORS = args.estimators
out = f"depth_{MAX_DEPTH}_estimators_{N_ESTIMATORS}"

# Generate an array of fake names
nCluster = 29
fake_arr = np.zeros((nCluster, nCluster))
columns_to_include = []
columns_names = []
for i in range(len(fake_arr)):
    for j in range(len(fake_arr[i])):
        if j > i:
            columns_to_include.append(nCluster*i+j)
            columns_names.append(f"{i+1}_{j+1}")
            
colorbar = Colors.define_RYG_colormap(bound_by_white=False)
X_train, X_weight, X_label, X_timestep = np.load("X_train.npy"), np.load("X_weight.npy"), np.load("X_label.npy"), np.load("X_timestep.npy").astype(int)
Y_train, Y_weight, Y_label, Y_timestep = np.load("Y_train.npy"), np.load("Y_weight.npy"), np.load("Y_label.npy"), np.load("Y_timestep.npy").astype(int)

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
Y_scaled = scaler.transform(Y_train)

def select_cv_idx(cv_group_labels, search):
    cvg = np.where(cv_group_labels == search)[0]
    train_set_idx = np.where(cv_group_labels != search)[0]
    return train_set_idx, cvg

cv_grps = []
for replica in range(ncv):
    train, test = select_cv_idx(X_timestep[:,0], replica)
    cv_grps.append((train,test))

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=0.001, max_features='sqrt', random_state=1, n_jobs=-1)
clf.fit(X_scaled, X_label, sample_weight=X_weight)

r = permutation_importance(clf, Y_scaled, Y_label, n_repeats=30, random_state=1, scoring='accuracy', n_jobs=-1)
importances = r["importances_mean"]
importances_dev = r["importances_std"]
reformat = []
for i in range(len(importances)):
    reformat.append([columns_names[i], np.abs(importances[i]), importances_dev[i]])

np.savetxt(f"importances_{out}.csv", np.array(reformat),fmt="%s")
