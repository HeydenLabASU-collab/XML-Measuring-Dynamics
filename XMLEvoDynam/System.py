import numpy as np
import glob
from XMLEvoDynam.MLInputTools import *

# ensure that dimensions are the same between systems

class System:
    def __init__(self, system_dir, systems, group1, group2, cutoff, input_file, index_file, output_file):
        self.system_dir = system_dir
        self.systems = systems
        self.group1 = group1
        self.group2 = group2
        self.column_names = MLInputTools.construct_feature_names(group1, group2)
        self.keeper_idx = []
        self.n_features = 0
        self.n_samples = 0
        self.threshold = lambda v: (np.array(v) < cutoff)
        self.input_file = input_file
        self.index_file = index_file
        self.output_file = output_file
        self.cutoff = cutoff

    def define_columns_within_cutoff(self):

        keeper = np.full(len(np.load(sorted(glob.glob(f"{self.system_dir}/{self.systems[0]}*npy"))[0], allow_pickle=True)[0]), False)
        keeper = keeper[:-3]

        for system in range(len(self.systems)):
            for file in sorted(glob.glob(f"{self.system_dir}/{self.systems[system]}*npy")):
                arr = np.load(file, allow_pickle=True, mmap_mode='r')
                for i in range(arr.shape[0]):
                    row = arr[i, :-3]
                    x = self.threshold(row)
                    keeper = np.logical_or(keeper, x)
                    self.n_samples += 1

        keeper_idx = []
        for decision in range(len(keeper)):
            if keeper[decision] == True and str(self.column_names[decision].split("_")[0]) != str(self.column_names[decision].split("_")[1]):
                keeper_idx.append(decision)

        keeper_idx.append(len(keeper))
        keeper_idx.append(len(keeper)+1)
        keeper_idx.append(len(keeper)+2)

        self.n_features = len(keeper_idx)
        self.keeper_idx = keeper_idx

        np.save(f"{self.index_file}", self.keeper_idx)

        return self.keeper_idx



    def generate_inputSet_within_cutoff(self):
        dataset = np.memmap(f'{self.input_file}', dtype='float32', mode='w+', shape=(self.n_samples, self.n_features))
        end = 0
        for system in range(len(self.systems)):
            for file in sorted(glob.glob(f"{self.system_dir}/{self.systems[system]}*npy")):
                arr = np.load(file, allow_pickle=True, mmap_mode='r')
                begin = end
                end = begin+len(arr)
                dataset[begin:end] = arr[:, self.keeper_idx]

        # Make sure data is flushed to disk
        dataset.flush()
        return dataset

    def output_system_info(self):
        dictionary = {'n_samples': self.n_samples,
                      'n_features': self.n_features,
                      'g1': self.group1,
                      'g2': self.group2,
                      'cutoff': self.cutoff,
                      'keeper_idx': self.keeper_idx,
                      'column_names': self.column_names,
                      'system_dir': self.system_dir,
                      'systems': self.systems,
                      'input_file': self.input_file,
                      'index_file': self.index_file,
                      }
        np.save(f'{self.output_file}', dictionary)
        return dictionary

    def load_system_from_dictionary(fIn):
        MLinfo = np.load(f"{fIn}", allow_pickle=True).item()
        return System(MLinfo['system_dir'], MLinfo['systems'], MLinfo['g1'], MLinfo['g2'], 6)

