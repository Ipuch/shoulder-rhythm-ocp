import numpy as np
import bioviz

try:
    import biorbd
except:
    import biorbd_casadi as biorbd

import pickle

# load model nb1

file1 = "../simulation_results/MuscleDriven_constraint_with_slack_test.pkl"
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

