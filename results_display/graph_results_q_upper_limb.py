import numpy as np
import bioviz
from scipy import stats
from math import ceil

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
import pickle


# load data
def get_exp_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
            except:
                break
    return data * 180 / np.pi


def get_created_q_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas = data_tmp["states"]["q"][:9, :] * 180 / np.pi
    return datas


def get_created_tau_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas1 = data_tmp["controls"]["tau"][:9, :]
    datas2 = np.zeros((1, 250))
    datas_formated = np.zeros((datas1.shape[0], datas2.shape[1]))
    for i in range(datas1.shape[0]):
        for j in range(datas1.shape[1]-1):
            for k in range(5):
                datas_formated[i, j*5+k] = datas1[i, j]
    datas2[0, :] = data_tmp["states"]["q"][6, :-1] * 180 / np.pi
    datas = np.concatenate([datas_formated, datas2])
    return datas


def get_created_muscles_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas1 = data_tmp["controls"]["muscles"]
    datas2 = np.zeros((1, 250))
    datas_formated = np.zeros((datas1.shape[0], datas2.shape[1]))
    for i in range(datas1.shape[0]):
        for j in range(datas1.shape[1]-1):
            for k in range(5):
                datas_formated[i, j*5+k] = datas1[i, j]
    datas2[0, :] = data_tmp["states"]["q"][6, :-1] * 180 / np.pi
    datas = np.concatenate([datas_formated, datas2])
    return datas


def add_datas_to_plot(abscissa: int, ordinates: np.ndarray, color: str, label: str, axs):
    c = 0
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            try:
                axs[i, j].plot(abscissa, ordinates[c, :], color=color)
                c += 1
            except:
                break

    axs[0, 0].plot(abscissa, ordinates[0, :],
                   color=color,
                   label=label
                   )


q_Clara = get_exp_data_from_pickle("q_exp_Clara.pkl")

q_file_2 = get_created_q_from_pickle(
    "UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_15_lever_bras_no_offset_no_slack_all_muscles_derivative_true_150000_1500.pkl")
q_file_3 = get_created_q_from_pickle(
    "UpperLimbMuscle_2_lig_200_range_2_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl")

# plot q
figX, axs = plt.subplots(3, 3)

add_datas_to_plot(q_Clara[6, :], q_Clara, color='b', label="experimental data", axs=axs)
add_datas_to_plot(q_file_2[6, :], q_file_2, label="file 2", color='g', axs=axs)
add_datas_to_plot(q_file_3[6, :], q_file_3, label="file 3", color='orange', axs=axs)

axs[0, 0].set_title("clav X")

axs[0, 1].set_title("clav Y")

axs[0, 2].set_title("scap X")

axs[1, 0].set_title("scap Y")

axs[1, 1].set_title("scap Z")

axs[1, 2].set_title("shoulder plan X")

axs[2, 0].set_title("shoulder ele Y")

axs[2, 1].set_title("shoulder rot Z")

figX.legend(loc="lower right",  # Position of legend
            borderaxespad=0,  # Small spacing around legend box
            title="Legend", prop={"size": 15})  # Title for the legend
figX.suptitle("evolution des q generalises au cours du mouvement, en degres")
figX.supxlabel('n_shooting')
figX.supylabel('angle in degres')

tau_old_and_hum_bounds = get_created_tau_from_pickle(
    "UpperLimbMuscle_2_lig_200_range_2_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl")
tau_new_bounds = get_created_tau_from_pickle(
    "UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_15_lever_bras_no_offset_no_slack_all_muscles_derivative_true_150000_1500.pkl")
tau_200000 = get_created_tau_from_pickle(
    "UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl")
# figure tau
figY, bxs = plt.subplots(3, 3)

add_datas_to_plot(tau_old_and_hum_bounds[9, :], tau_old_and_hum_bounds, label="file 2", color='b', axs=bxs)
add_datas_to_plot(tau_new_bounds[9, :], tau_new_bounds, label="file 3", color='orange', axs=bxs)
add_datas_to_plot(tau_200000[9, :], tau_200000, label="200000", color='r', axs=bxs)

bxs[0, 0].set_title("clav X")
bxs[0, 1].set_title("clav Y")
bxs[0, 2].set_title("scap X")
bxs[1, 0].set_title("scap Y")
bxs[1, 1].set_title("scap Z")
bxs[1, 2].set_title("shoulder plan X")
bxs[2, 0].set_title("shoulder ele Y ")
bxs[2, 1].set_title("shoulder rot Z")

figY.legend(loc="lower right",  # Position of legend
            borderaxespad=0,  # Small spacing around legend box
            title="Legend", prop={"size": 15})  # Title for the legend
figY.suptitle("evolution des tau au cours du mouvement")
figY.supxlabel('n_shooting')
figY.supylabel('tau')

muscles_150000_controles = get_created_muscles_from_pickle(
    'UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_15_lever_bras_no_offset_no_slack_all_muscles_derivative_true_150000_1500.pkl')
muscles_1500_controles = get_created_muscles_from_pickle(
    'UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl')
muscles_range2_1500_controles = get_created_muscles_from_pickle(
    "UpperLimbMuscle_2_lig_200_range_2_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl")
model = biorbd.Model(
    "/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/models/wu_converted_definitif_without_floating_base.bioMod")
if len(muscles_150000_controles) <= 4:
    n_column = 1
else:
    n_column = ceil(len(muscles_150000_controles) / 4)

figZ, cxs = plt.subplots(4, n_column)
c = 0
for i in range(4):
    for j in range(n_column):
        try:
            cxs[i, j].plot(muscles_150000_controles[31, :], muscles_150000_controles[c], label='file1')
            cxs[i, j].plot(muscles_1500_controles[31, :], muscles_1500_controles[c], label='file2')
            cxs[i, j].plot(muscles_range2_1500_controles[31, :], muscles_range2_1500_controles[c], label='file3')
            cxs[i, j].set_title(model.muscleNames()[c].to_string())
            cxs[i, j].set_ylim([-0.01, 1.01])
        except:
            break
        c += 1

figZ.legend(loc="lower right",  # Position of legend
            borderaxespad=0,  # Small spacing around legend box
            title="Legend", prop={"size": 15})  # Title for the legend
plt.subplots_adjust(top=0.945,
                    bottom=0.095,
                    left=0.075,
                    right=0.945,
                    hspace=0.285,
                    wspace=0.21)

plt.show()
