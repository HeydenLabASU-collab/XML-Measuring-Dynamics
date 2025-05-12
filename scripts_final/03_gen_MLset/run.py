from XMLEvoDynam.System import *
from XMLEvoDynam.MLStart import *

# Point to the dictionary containing system info
MLinfo = "../02_gen_inputset/MLinfo.npy"

# Load the system from information in the dictionary
epitope_system = System.load_system_from_dictionary(MLinfo)

# Output from step 2
epitope_system.input_file = "../02_gen_inputset/input.npy"
epitope_system.index_file = "../02_gen_inputset/keeper.npy"

# Generate train/test split given a test size (ratio)
ml = MLStart(epitope_system, 0.2)
ml.define_train_test_groups()
ml.export_ML_input(scalar_fit=False)
