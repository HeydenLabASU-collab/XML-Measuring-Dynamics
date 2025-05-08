import numpy as np
import glob
from XMLEvoDynam.MLInputTools import *
from XMLEvoDynam.System import *
from XMLEvoDynam.MLStart import *

MLinfo = np.load("../02_gen_inputset/MLinfo.npy", allow_pickle=True).item()
epitope_system = System.load_system_from_dictionary(MLinfo)
ml = MLStart(epitope_system, 0.2)
ml.define_train_test_groups()
ml.export_ML_input(scalar_fit=False)