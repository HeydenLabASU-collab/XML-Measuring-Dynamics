import numpy as np
from XMLEvoDynam.System import *

GROUP1 = np.arange(0,391,1)
GROUP2 = np.arange(391,405,1)
column_names = MLInputTools.construct_feature_names(GROUP1, GROUP2)

SYSTEM_DIR = "../01_prep"
SYSTEM = ["VGSDWRFLRGYHQYQ", "VGSDWRFLRGYHQYA"]
CUTOFF_DISTANCE = 6
INPUT_FILE = "input.npy"
INDEX_FILE = "keeper.npy"
OUTPUT_FILE = "MLinfo.npy"

epitope_system = System(SYSTEM_DIR, SYSTEM, GROUP1, GROUP2, CUTOFF_DISTANCE, INPUT_FILE, INDEX_FILE, OUTPUT_FILE)
keeper_idx = epitope_system.define_columns_within_cutoff()
dataset = epitope_system.generate_inputSet_within_cutoff()
data_info = epitope_system.output_system_info()

