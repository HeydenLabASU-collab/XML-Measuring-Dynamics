import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class MLStart:
    """
    A class for generating a train/test dataset splits from a provided System. Used in conjunction with System class.

    Attributes:
        system (list): List of filenames containing names of each system. See MLInputTools for more info.
        split_percent (float): Percentage (0-1) of data to use in the test set
        n_states (int): Number of systems
        n_samples_per_structure (int): Number of samples in each system
        X_train (np.array): Training feature set
        X_test (np.array): Test feature set
        y_train (np.array): Training labels
        y_test (np.array): Test labels
        W_train (np.array): Training sample weights
        W_test (np.array): Test sample weights
    """
    def __init__(self, SYSTEM, split_percent):
        self.system = SYSTEM
        self.split_percent = split_percent
        self.n_states = len(SYSTEM.systems)
        self.n_samples_per_structure = int(SYSTEM.n_features/len(SYSTEM.systems))
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.W_train = None
        self.W_test = None



    def define_train_test_groups(self):
        """
        Generate train/test splits with sklearn.

        Args:
            self (MLStart): MLStart object containing info on how to split the data
        """

        # Load unified dataset generated from System class
        dataset = np.memmap(f"{self.system.input_file}", dtype='float32', mode='r', shape=(self.system.n_samples, self.system.n_features))

        # X contains the features AND the weights
        X = dataset[:, :-2]

        # Labels
        y = dataset[:, -2]

        # Determine train and test splits
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.split_percent, random_state=42)

        # The last column in the feature set are the weights
        # Store the splits
        self.W_train = X_train[:,-1]
        self.X_train = X_train[:,:-1]
        self.W_test = X_test[:,-1]
        self.X_test = X_test[:,:-1]
        self.y_train = y_train
        self.y_test = y_test
        dataset.flush()

    def export_ML_input(self, scalar_fit=False):

        # Save the labels and the weights
        np.save('y_train.npy', self.y_train.astype(int))
        np.save('y_test.npy', self.y_test.astype(int))
        np.save('weight_train.npy', self.W_train)
        np.save('weight_test.npy', self.W_test)

        # IF a standard scalar fit is desired, apply the transformation and then export the features
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

        # No StandardScalar -> Just output the features
        else:
            np.save("xscaled_train.npy", self.X_train)
            np.save("xscaled_test.npy", self.X_test)
            print(f"FINAL SIZE:\nTrain Data:{np.shape(self.X_train)}\nTest Data:{np.shape(self.X_test)}\nTraining Label:{np.shape(self.y_train)}\nTest Labels:{np.shape(self.y_test)}\nWeight Train Data:{np.shape(self.W_train)}\nWeight Test Data:{np.shape(self.W_test)}")
