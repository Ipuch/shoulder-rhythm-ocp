import numpy as np
import bioviz
from scipy import stats

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
import pickle

# load model nb1

file1 = "/home/mickaelbegon/Documents/stage_nicolas/shoulder-rhythm-ocp/humerus_elevation/UpperLimbMuscle_2_lig_200_index_1_0.5_tau_low02_15_and_1500_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000_qdot_50_new_obj_1.6.pkl"
result_mat1 = {}
with open(file1, "rb") as f:
    while True:
        try:
            data_tmp = pickle.load(f)
            key = list(data_tmp.keys())[0]
            result_mat1[key] = data_tmp[key]
        except:
            break
#
model = biorbd.Model(
    "/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/models/wu_converted_definitif_without_floating_base.bioMod")

b = bioviz.Viz(loaded_model=model,
               show_muscles=False,
               show_floor=False,
               show_local_ref_frame=True,
               show_global_ref_frame=True,
               show_gravity_vector=True,
               )
all_q = np.zeros((model.nbQ(), 251))
all_q = data_tmp["states"]["q"]
b.load_movement(all_q)
b.exec()

