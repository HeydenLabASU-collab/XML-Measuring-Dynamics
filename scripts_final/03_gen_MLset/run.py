import numpy as np
import glob
from XMLEvoDynam.MLInputTools import *
from XMLEvoDynam.System import *
from XMLEvoDynam.MLStart import *

MLinfo = "../02_gen_inputset/MLinfo.npy"
epitope_system = System.load_system_from_dictionary(MLinfo)
epitope_system.input_file = "../02_gen_inputset/input.npy"
epitope_system.index_file = "../02_gen_inputset/keeper.npy"
ml = MLStart(epitope_system, 0.2)
ml.define_train_test_groups()
ml.export_ML_input(scalar_fit=False)
