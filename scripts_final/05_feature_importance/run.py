from XMLEvoDynam.MLInputTools import *
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

GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)
print(f"Generating Column Names")
column_names = MLInputTools.construct_feature_names(GROUP1, GROUP2)

keeper = np.load(f"../02_gen_inputset/keeper.npy")
keeper = keeper[:-3]

column_names = column_names[keeper]

MAX_DEPTH = args.depth
N_ESTIMATORS = args.estimators
MAX_DEPTH = 5
N_ESTIMATORS = 5
out = f"depth_{MAX_DEPTH}_estimators_{N_ESTIMATORS}"
print("Get files")            
colorbar = Colors.define_RYG_colormap(bound_by_white=False)

X_scaled = np.load('../03_gen_MLset/xscaled_train.npy', mmap_mode='r')
Y_train_export = np.load('../03_gen_MLset/y_train.npy', allow_pickle=True)
Weight_train_export = np.load('../03_gen_MLset/weight_train.npy', allow_pickle=True)
print("Files in")

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, min_samples_leaf=0.001, max_features='sqrt', random_state=1, n_jobs=-1)
clf.fit(X_scaled, Y_train_export, sample_weight=Weight_train_export)
print("fit")
r = permutation_importance(clf, X_scaled, Y_train_export, n_repeats=30, random_state=1, scoring='accuracy', n_jobs=-1)
print("done")
importances = r["importances_mean"]
importances_dev = r["importances_std"]
reformat = []
for i in range(len(importances)):
    reformat.append([column_names[i], np.abs(importances[i]), importances_dev[i]])

np.savetxt(f"importances_{out}.csv", np.array(reformat),fmt="%s")
