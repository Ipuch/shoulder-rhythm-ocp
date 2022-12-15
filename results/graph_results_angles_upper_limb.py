import numpy as np

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import pickle
from ocp.utils import axemobile_yxy, rot_mat_to_euler


def get_exp_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data = pickle.load(f)
            except:
                break
    return data


def get_created_data_from_pickle(file: str):
    with open(file, "rb") as f:
        while True:
            try:
                data_tmp = pickle.load(f)
            except:
                break
    datas = data_tmp["states"]["q"][:9, :]
    # datas = data_tmp
    return datas


def from_q_to_shoulder_euler_angles(datas: np.ndarray, start_from_zero: bool = False):
    """
    :param datas: generalized q of the model
    :param start_from_zero: if we put the values of the beginning of the motion at 0 or not
    :return:
    values in radiant of the elevation of the humerus in regard of the thorax, and SC, AC, GL and scapulothoracic
    angles evolution
    """

    generalized_q = np.zeros((10, 251))
    generalized_q[:9, :] = datas

    theta_euler = np.zeros((datas.shape[1]))
    sternoclavicular = np.zeros((3, datas.shape[1]))
    scapulothoracic = np.zeros((3, datas.shape[1]))
    acromioclavicular = np.zeros((3, datas.shape[1]))
    glenohumeral = np.zeros((3, datas.shape[1]))
    unwrapped_gh1 = np.zeros((1, datas.shape[1]))
    unwrapped_gh2 = np.zeros((1, datas.shape[1]))
    unwrapped_gh3 = np.zeros((1, datas.shape[1]))
    unwrapped_gh = np.zeros((3, datas.shape[1]))

    for i in range(datas.shape[1]):
        generalized_q1_i = generalized_q[:, i]
        generalized_q_0 = generalized_q[:, 0]
        rot_matrix = np.zeros((3, 3, len(segment_list)))
        rot_matrix_init = np.zeros((3, 3, len(segment_list)))
        thorax_init_position = model.globalJCS(generalized_q_0, 2).rot().to_array()

        # actual and initial position of segments
        for j, s in enumerate(segment_list):
            rot_matrix_init[:, :, j] = model.globalJCS(generalized_q_0, s).rot().to_array()
            rot_matrix[:, :, j] = model.globalJCS(generalized_q1_i, s).rot().to_array()

        # actual rotation matrix of segments
        clav1_thorax_rotation = thorax_init_position.T @ rot_matrix[:, :, 0]  # rotation matrix of SC
        scap1_thorax_rotation = thorax_init_position.T @ rot_matrix[:, :, 1]  # rotation matrix of ST
        hum1_thorax_rotation = thorax_init_position.T @ rot_matrix[:, :, 2]  # rotation matrix of TH
        scap1_clav1_rotation = rot_matrix[:, :, 0].T @ rot_matrix[:, :, 1]  # rotation matrix of AC
        hum1_scap1_rotation = rot_matrix[:, :, 1].T @ rot_matrix[:, :, 2]  # rotation matrix of GH

        if start_from_zero:
            # initial rotation matrix of segments
            clav1_thorax_rotation_0 = thorax_init_position.T @ rot_matrix_init[:, :, 0]  # rot matrix of SC init pos
            scap1_thorax_rotation_0 = thorax_init_position.T @ rot_matrix_init[:, :, 1]  # rot matrix of ST init pos
            hum1_thorax_rotation_0 = thorax_init_position.T @ rot_matrix_init[:, :, 2]  # rot matrix of TH init pos
            scap1_clav1_rotation_0 = rot_matrix_init[:, :, 0].T @ rot_matrix_init[:, :, 1]  # rot matrix of AC init pos
            hum1_scap1_rotation_0 = rot_matrix_init[:, :, 1].T @ rot_matrix_init[:, :, 2]  # rot matrix of GH init pos

            # rotation matrix between init to actual position
            SC1_rotation_matrix = clav1_thorax_rotation_0.T @ clav1_thorax_rotation
            ST1_rotation_matrix = scap1_thorax_rotation_0.T @ scap1_thorax_rotation
            TH1_rotation_matrix = hum1_thorax_rotation_0.T @ hum1_thorax_rotation
            AC1_rotation_matrix = scap1_clav1_rotation_0.T @ scap1_clav1_rotation
            GH1_rotation_matrix = hum1_scap1_rotation_0.T @ hum1_scap1_rotation

            # rotation matrix to euler vectors
            rot1_SC_euler = rot_mat_to_euler(SC1_rotation_matrix, seq="yxz")
            rot1_ST_euler = rot_mat_to_euler(ST1_rotation_matrix, seq="yxz")
            rot1_hum_euler = axemobile_yxy(TH1_rotation_matrix)
            rot1_AC_euler = rot_mat_to_euler(AC1_rotation_matrix, seq="yxz")
            rot1_GH_euler = axemobile_yxy(GH1_rotation_matrix)

            sternoclavicular[0, i] = rot1_SC_euler[0]
            sternoclavicular[1, i] = rot1_SC_euler[1]
            sternoclavicular[2, i] = rot1_SC_euler[2]

            scapulothoracic[0, i] = rot1_ST_euler[0]
            scapulothoracic[1, i] = rot1_ST_euler[1]
            scapulothoracic[2, i] = rot1_ST_euler[2]

            acromioclavicular[0, i] = rot1_AC_euler[0]
            acromioclavicular[1, i] = rot1_AC_euler[1]
            acromioclavicular[2, i] = rot1_AC_euler[2]

            glenohumeral[0, i] = rot1_GH_euler[0]
            glenohumeral[1, i] = rot1_GH_euler[1]
            glenohumeral[2, i] = rot1_GH_euler[2]

            theta_euler[i] = rot1_hum_euler[1]

        else:
            #  rotation matrix to euler vectors
            rot1_SC_euler = rot_mat_to_euler(clav1_thorax_rotation, seq="yxz")
            rot1_ST_euler = rot_mat_to_euler(scap1_thorax_rotation, seq="yxz")
            rot1_hum_euler = axemobile_yxy(hum1_thorax_rotation)
            rot1_AC_euler = rot_mat_to_euler(scap1_clav1_rotation, seq="yxz")
            rot1_GH_euler = axemobile_yxy(hum1_scap1_rotation)

            #  rotation matrix to euler vectors

            sternoclavicular[0, i] = rot1_SC_euler[0]
            sternoclavicular[1, i] = rot1_SC_euler[1]
            sternoclavicular[2, i] = rot1_SC_euler[2]

            scapulothoracic[0, i] = rot1_ST_euler[0]
            scapulothoracic[1, i] = rot1_ST_euler[1]
            scapulothoracic[2, i] = rot1_ST_euler[2]

            acromioclavicular[0, i] = rot1_AC_euler[0]
            acromioclavicular[1, i] = rot1_AC_euler[1]
            acromioclavicular[2, i] = rot1_AC_euler[2]

            glenohumeral[0, i] = rot1_GH_euler[0]
            glenohumeral[1, i] = rot1_GH_euler[1]
            glenohumeral[2, i] = rot1_GH_euler[2]

            theta_euler[i] = rot1_hum_euler[1]

    unwrapped_gh1 = np.unwrap(glenohumeral[0])
    unwrapped_gh2 = np.unwrap(glenohumeral[1])
    unwrapped_gh3 = np.unwrap(glenohumeral[2])

    unwrapped_gh[0, :], unwrapped_gh[1, :], unwrapped_gh[2, :] = unwrapped_gh1, unwrapped_gh2, unwrapped_gh3
    values_dict = {
        'sternoclavicular': sternoclavicular,
        'scapulothoracic': scapulothoracic,
        'acromioclavicular': acromioclavicular,
        'glenohumeral': unwrapped_gh
    }
    return theta_euler, values_dict


def add_datas_to_plot(abscissa: np.ndarray, values: dict, color: str, label: str, axs):
    keys = list(values.keys())

    # for i, key in enumerate(keys):
    for i, (key, value) in enumerate(values.items()):
        for j, val in enumerate(value):
            axs[i, j].plot(abscissa[:] * 180 / np.pi, val * 180 / np.pi, color=color)

    axs[0, 0].plot(abscissa[:] * 180 / np.pi, values[keys[0]][0, :] * 180 / np.pi,
                   color=color,
                   label=label
                   )


# get data from pickle files
data_Clara = get_exp_data_from_pickle("../datas/q_exp_Etienne.pkl")
data_file_1 = get_created_data_from_pickle(
    "../datas/UpperLimbMuscle_default_no_constraint_test_tau150_q100.pkl")

model = biorbd.Model(
    "/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/models/wu_converted_definitif_without_floating_base.bioMod")

segment_list = [8, 15, 22]  # in order : clavicle, scapula, humerus

hum_elevation_Clara, angles_Clara = from_q_to_shoulder_euler_angles(data_Clara)
hum_elevation_file1, angles_file1 = from_q_to_shoulder_euler_angles(data_file_1)
Matsurama_datas = np.zeros((3, 10))  # in order :STProtracElev, STMedRot, STTiltlevation
Matsurama_abscissa = np.zeros((3, 10))
Matsurama_datas[1, :] = [-9.3376, -12.5557, -16.169, -19.7828, -23.3396, -26.4445, -29.9449, -33.3324, -37.2287, -40.8424]
Matsurama_abscissa[1, :] = [30.088, 39.561, 50.088, 59.649, 70.175, 79.825, 90, 99.912, 109.912, 119.825]
Matsurama_datas[2, :] = [-8.4559, -7.9246, -7.2963, -6.7164, -5.9667, -4.9503, -4.0795, -3.0874, -1.9739, -0.5452]
Matsurama_abscissa[2, :] = [29.946, 40.091, 50.148, 60.206, 69.826, 79.797, 90.031,100.176, 109.798, 119.596]
Matsurama_datas[0, :] = [-6.2439, -5.5854, -5.0976, -4.561, -3.9512, -3.1463, -2.3415, -1.878, -0.878, 0.3415]
Matsurama_abscissa[0, :] = [30.311, 40.396, 50.132, 60.393, 70.392, 80.389, 90.124, 99.772, 109.944, 120.028]
figX, axs = plt.subplots(4, 3)

add_datas_to_plot(hum_elevation_Clara, angles_Clara, 'b', 'Clara', axs)
add_datas_to_plot(hum_elevation_file1, angles_file1, 'r', 'file 1', axs)

for m in range(3):
    axs[1, m].plot(Matsurama_abscissa[m, :], Matsurama_datas[m, :], color='gray')

axs[0, 0].set_ylabel("Sternoclavicular \n Retraction (Degrees) \n Protraction(+)/Retraction(-)")
axs[0, 1].set_ylabel("Sternoclavicular \n Elevation (Degrees) \n Depression(+)/Elevation(-)")
axs[0, 2].set_ylabel("Sternoclavicular \n Rotation (Degrees) \n Top Backwards(+)/Forwards(-)")

axs[1, 0].set_ylabel("Scapulothoracic \n Retraction (Degrees)\n Protraction(+)/Retraction(-) ")
axs[1, 1].set_ylabel("Scapulothoracic \n Rotation (Degrees) \n Medial(+)/Lateral(-)")
axs[1, 2].set_ylabel("Scapulothoracic \n Tilting (Degrees) \n Anterior(+)/Posterior(-)")

axs[2, 0].set_ylabel("Acromioclavicular \n Internal Rotation (Degrees) \n AC-retraction(+)/AC-protraction(-)")
axs[2, 1].set_ylabel("Acromioclavicular \n Rotation (Degrees) \n AC-Medial(+)/AC-Lateral(-)")
axs[2, 2].set_ylabel("Acromioclavicular \n Tilting (Degrees) \n AC-Posterior(+)/AC-Anterior(-)")

axs[3, 0].set_ylabel("Glenohumeral \n Plane of Elevation (Degrees)")
axs[3, 1].set_ylabel("Glenohumeral \n Elevation (Degrees) \n Depression(+)/Elevation(-)")
axs[3, 2].set_ylabel("Glenohumeral \n Axial Rotation (Degrees) \n Internal(+)/External")

for i in range(2):
    for j in range(3):
        axs[i, j].set_xlabel("Humerothoracic Elevation (Degrees)")  # abscissa for all subplots

figX.suptitle("Shoulder rotations")
figX.legend(loc="upper right",  # Position of legend
            borderaxespad=0,  # Small spacing around legend box
            title="Legend", prop={"size": 15})

plt.subplots_adjust(top=0.93,
                    bottom=0.075,
                    left=0.09,
                    right=0.91,
                    hspace=0.35,
                    wspace=0.28)

plt.show()
