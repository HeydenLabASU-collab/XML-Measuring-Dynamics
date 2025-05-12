import numpy as np
from XMLEvoDynam.System import *

# Example Groups
GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)

# Generate column names from group idxs
column_names = MLInputTools.construct_feature_names(GROUP1, GROUP2)

SYSTEM_DIR = "../01_prep" # Directory containing features generated from trajectories
SYSTEM = ["VGSDWRFLRGYHQYQ", "VGSDWRFLRGYHQYA"] # filename prefixes for each of the systems ran in step 1
CUTOFF_DISTANCE = 6 # Cutoff distance used to keep/discard features
INPUT_FILE = "input.npy" # output file containing features from all SYSTEMs (input for step 3)
INDEX_FILE = "keeper.npy" # output file containing column idxs of features within the cutoff (input for step 3)
OUTPUT_FILE = "MLinfo.npy" # output file containing dictionary to store info about SYSTEMs

# Generate a new "System" which contains information about all trajectories
epitope_system = System(SYSTEM_DIR, SYSTEM, GROUP1, GROUP2, CUTOFF_DISTANCE, INPUT_FILE, INDEX_FILE, OUTPUT_FILE, 0, 0)

# For each feature (residue pairs), figure out which columns contain at least one sample (timestep) within the cutoff
keeper_idx = epitope_system.define_columns_within_cutoff()

# Given the indeces of columns to keep, generate ONE file containing all samples from all systems
dataset = epitope_system.generate_inputSet_within_cutoff()

# Output a dictionary containing information about the SYSTEMs
data_info = epitope_system.output_system_info()

