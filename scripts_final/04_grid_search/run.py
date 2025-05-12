from XMLEvoDynam.ML import *

# Number of cross-validation sets to use
ncv=4

# Number of trees in the forest (list)
n_estimators = [5,10,25,50,100]

# Depth of each tree in the forest (list)
max_depth = [5,10,20]

# Point to the training set features
X = '../03_gen_MLset/xscaled_train.npy'

# Point to the training set labels
y = '../03_gen_MLset/y_train.npy'

# Point to the training set weights
w = '../03_gen_MLset/weight_train.npy'

# Run grid search over forest size (n_estimators) and tree depth (max_depth)
clf = ML.run_grid_search(ncv, n_estimators, max_depth, X, y, w, min_samples_leaf = 0.01, n_jobs=1)

# Generate visualization of the grid search results
ML.generate_grid_search_plot(clf, max_depth, n_estimators, "epitope", min_samples_leaf = 0.01)