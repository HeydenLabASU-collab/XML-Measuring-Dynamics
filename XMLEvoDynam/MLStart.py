import numpy as np
from sklearn.model_selection import train_test_split

class MLStart:

    def __init__(self, SYSTEM, split_percent):
        self.system = SYSTEM
        self.split_percent = split_percent
        self.n_states = len(self.system['systems'])
        self.n_samples_per_structure = int(self.system['n_features']/self.n_states)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.W_train = None
        self.W_test = None


    def define_train_test_groups(self):
        dataset = np.memmap(f"{self.system['input_file']}", dtype='float32', mode='r', shape=(self.system['n_samples'], self.system['n_features']))
        X = dataset[:, :-2]
        y = dataset[:, -2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_percent, random_state=42)
        self.W_train = X_train[:,-1]
        self.X_train = X_train[:,:-1]
        self.W_test = X_test[:,-1]
        self.X_test = X_test[:,:-1]
        self.y_train = y_train
        self.y_test = y_test
        dataset.flush()

    def export_ML_input(self, scalar_fit=False):
        np.save('y_train.npy', self.y_train.astype(int))
        np.save('y_test.npy', self.y_test.astype(int))
        np.save('weight_train.npy', self.W_train)
        np.save('weight_test.npy', self.W_test)

        if scalar_fit is True:
            print("Fir Standard Scalar to Train Set")
            scaler = preprocessing.StandardScaler().fit(self.X_train)

            print("Apply Transformation to Train Set")
            Xscaleset_train = scaler.transform(self.X_train)

            print("Apply Transformation to Test Set")
            Xscaleset_test = scaler.transform(self.X_test)


            np.save("xscaled_train.npy", Xscaleset_train)
            np.save("xscaled_test.npy", Xscaleset_test)
            print(f"FINAL SIZE:\nTrain Data:{np.shape(Xscaleset_train)}\nTest Data:{np.shape(Xscaleset_test)}\nTraining Label:{np.shape(self.y_train)}\nTest Labels:{np.shape(self.y_test)}\nWeight Train Data:{np.shape(self.W_train)}\nWeight Test Data:{np.shape(self.W_test)}")
        else:
            np.save("xscaled_train.npy", self.X_train)
            np.save("xscaled_test.npy", self.X_test)
            print(f"FINAL SIZE:\nTrain Data:{np.shape(self.X_train)}\nTest Data:{np.shape(self.X_test)}\nTraining Label:{np.shape(self.y_train)}\nTest Labels:{np.shape(self.y_test)}\nWeight Train Data:{np.shape(self.W_train)}\nWeight Test Data:{np.shape(self.W_test)}")
