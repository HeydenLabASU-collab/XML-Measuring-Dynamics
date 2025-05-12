from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ML:
    def run_grid_search(ncv, n_estimators, max_depth, X, y, w, min_samples_leaf=0.01, n_jobs=1):
        parameters = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': [min_samples_leaf]}
        X = np.load(f'{X}', mmap_mode='r')
        y = np.load(f'{y}', allow_pickle=True)
        w = np.load(f'{w}', allow_pickle=True)
        svc = RandomForestClassifier(max_features='sqrt', random_state=1)
        clf = GridSearchCV(svc, parameters, scoring='accuracy', verbose=10, n_jobs=n_jobs, return_train_score=True)
        clf.fit(X, y, sample_weight=w)
        return clf


    def generate_grid_search_plot(clf, max_depth, n_estimators, output_name, min_samples_leaf=0.01):
        df = pd.concat([pd.DataFrame(clf.cv_results_["params"]),
                        pd.DataFrame(clf.cv_results_["mean_train_score"], columns=["Train Accuracy"]),
                        pd.DataFrame(clf.cv_results_["std_train_score"], columns=["Train STD"]),
                        pd.DataFrame(clf.cv_results_["mean_test_score"], columns=["Test Accuracy"]),
                        pd.DataFrame(clf.cv_results_["std_test_score"], columns=["Test STD"])], axis=1)
        df_query = df[(df["min_samples_leaf"] == min_samples_leaf)]
        train_accuracy_arr = np.array(df_query["Train Accuracy"]).reshape((len(max_depth), len(n_estimators)))
        test_accuracy_arr = np.array(df_query["Test Accuracy"]).reshape((len(max_depth), len(n_estimators)))

        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        im = ax[0].imshow(train_accuracy_arr)
        im = ax[1].imshow(test_accuracy_arr)

        # Show all ticks and label them with the respective list entries
        for p in range(2):
            ax[p].set_yticks(range(len(max_depth)), labels=max_depth, rotation=45, ha="right", rotation_mode="anchor")
            ax[p].set_xticks(range(len(n_estimators)), labels=n_estimators)

            # Loop over data dimensions and create text annotations.
            for i in range(len(max_depth)):
                for j in range(len(n_estimators)):
                    if p == 0:
                        text = ax[p].text(j, i, f"{np.round(train_accuracy_arr[i, j], 3)}", ha="center", va="center",
                                          color="b")
                        ax[p].set_title("Training Data Grid Search")
                    if p == 1:
                        text = ax[p].text(j, i, f"{np.round(test_accuracy_arr[i, j], 3)}", ha="center", va="center",
                                          color="b")
                        ax[p].set_title("Test Data Grid Search")
            ax[p].set_ylabel("Max Depth")
            ax[p].set_xlabel("Number of Estimators")
        fig.tight_layout()

        df.to_csv(f"{output_name}_gridsearch.csv", index=False)
        np.savetxt(f"{output_name}_training_accuracy.csv", train_accuracy_arr)
        np.savetxt(f"{output_name}_test_accuracy.csv", test_accuracy_arr)
        plt.savefig(f"{output_name}.jpg", dpi=400)

#ncv=4
#replicalst = np.array([[15, 12, 9, 0],[10, 1, 6, 3],[8, 5, 7, 13],[14, 2, 11, 4]])
#cv_grps = []
# Define REPLICA numbers
#zero_sequence = np.repeat(0, 10000)
#one_sequence = np.repeat(1, 10000)
#arr = np.concatenate([zero_sequence, one_sequence])
#arr = arr.reshape(len(arr),1)
#
#def select_cv_lst(cv_group_labels, search):
#    cvg = []
#    train_set_idx = []
#    for dt in range(len(cv_group_labels)):
#        if cv_group_labels[dt] in search:
#            cvg.append(dt)
#        elif cv_group_labels[dt] not in search:
#            train_set_idx.append(dt)
#    return np.array(train_set_idx), np.array(cvg)
#
#for replica in range(ncv):
#    train, test = select_cv_lst(arr[:,0], list(replicalst[replica]))
#    cv_grps.append((train,test))