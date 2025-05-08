import numpy as np
import glob, os
import matplotlib.pyplot as plt
from XMLEvoDynam.MLInputTools import *


# ensure that dimensions are the same between systems

def define_columns_within_cutoff(system_dir, systems, column_names, fOut, CUTOFF_DISTANCE):
    threshold = lambda v: (np.array(v) < CUTOFF_DISTANCE)
    keeper = np.full(len(np.load(sorted(glob.glob(f"{system_dir}/{systems[0]}*npy"))[0], allow_pickle=True)[0]), False)
    keeper = keeper[:-3]
    
    n_samples = 0
    for system in range(len(systems)):
        for file in sorted(glob.glob(f"{system_dir}/{systems[system]}*npy")):
            arr = np.load(file, allow_pickle=True, mmap_mode='r')
            for i in range(arr.shape[0]):
                row = arr[i, :-3]
                x = threshold(row)
                keeper = np.logical_or(keeper, x)
                n_samples += 1
        
    keeper_idx = []
    for decision in range(len(keeper)):
        if keeper[decision] == True and str(column_names[decision].split("_")[0]) != str(column_names[decision].split("_")[1]):
            keeper_idx.append(decision)
            
    keeper_idx.append(len(keeper))
    keeper_idx.append(len(keeper)+1)
    keeper_idx.append(len(keeper)+2)
    n_features = len(keeper_idx)
    if fOut != None:
        np.save(f"{fOut}", keeper_idx)
        
    return keeper_idx, n_samples, n_features
    


def generate_inputSet_within_cutoff(fIn, keeper_idx, n_samples, n_features, system_dir, systems):
    dataset = np.memmap(f'{fIn}', dtype='float32', mode='w+', shape=(n_samples, n_features))
    end = 0
    for system in range(len(SYSTEM)):
        for file in sorted(glob.glob(f"{SYSTEM_DIR}/{SYSTEM[system]}*npy")):
            arr = np.load(file, allow_pickle=True, mmap_mode='r')
            begin = end
            end = begin+len(arr)
            dataset[begin:end] = arr[:, keeper_idx]
    
    # Make sure data is flushed to disk
    dataset.flush()
    return dataset

def output_ML_info(n_samples, n_features, keeper_idx, fOut):
    dictionary = {'n_samples':n_samples,'n_features':n_features,'keeper_idx':keeper_idx}
    np.save(f'{fOut}', dictionary) 

GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)
column_names = MLInputTools.construct_feature_names(GROUP1, GROUP2)

SYSTEM = ["VGSDWRFLRGYHQYQ", "VGSDWRFLRGYHQYA"]
SYSTEM_DIR = "../01_prep"
CUTOFF_DISTANCE = 6

keeper_idx, n_samples, n_features = define_columns_within_cutoff(SYSTEM_DIR, SYSTEM, column_names, "keeper.npy", CUTOFF_DISTANCE)
generate_inputSet_within_cutoff("input.npy", keeper_idx, n_samples, n_features, SYSTEM_DIR, SYSTEM)
output_ML_info(n_samples, n_features, keeper_idx, "MLinfo.npy")

