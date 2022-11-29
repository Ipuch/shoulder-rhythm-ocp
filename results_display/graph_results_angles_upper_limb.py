import numpy as np

try:
    import biorbd
except:
    import biorbd_casadi as biorbd
import os
import matplotlib.pyplot as plt
import pickle


def axemobile_yxy(matrix: np.ndarray):
    """

    Get Euler angles from a rotation matrix

    Composed rotation matrices for all the composition orders:
    Ryxy = [c(a1) * c(a3) - s(a1) * c(a2) * s(a3)     s(a1) * s(a2)    s(a1) * c(a3) + s(a1) * c(a2) * c(a3)]
           [s(a2) * s(a3)                             c(a2)            - s(a2) * c(a3)]
           [-c(a3) * s(a1) - c(a1) * c(a2) * s(a3)    c(a1) * s(a2)    c(a1) * c(a2) * c(a3) - s(a1) * s(a3)]

    With a1 = alpha = rotation around mobile y axis
         a2 = beta = rotation around mobile x' axis
         a3 = gamma = rotation around mobile y'' axis

    All the matrices have a similar form, in which:
        *The rotation around the second axis can be accessed through its sine
        *The  other  two  can  be  accessed through their tangents

    So all we need is a list of where to find the necessary data in the composed
    matrix, and whether we need to swap the sign of each element before taking
    the inverse trigonometric function.

    Get back the angles around mobile axes for a rotation matrix
    """

    OutputMatrix = np.zeros((3, 3))
    Matrix = matrix

    a1 = 0
    a2 = np.arccos(Matrix[1, 1])  # Retrieve the second angle in the sequence by its sinus
    a3 = 0

    # Retrieve the first angle in the sequence by its tangent
    if (Matrix[0, 1] != 0) & (Matrix[2, 1] != 0):
        # General case, tan(ay) = FNumerator / FDenominator
        a1 = np.arctan2(Matrix[0, 1], Matrix[2, 1])
    elif (Matrix[0, 1] == 0) & (Matrix[2, 1] == 0):
        # Case where cos(ax) = 0, and the other two angles are not defined
        a1 = 0
    elif (Matrix[0, 1] == 0) & (Matrix[2, 1] != 0):
        # Case where sin(ay) = 0 so ay is 0° or 180°
        # Ratio of FNumerator to cos(ax) is cos(ay)
        a1 = np.real(np.arccos(Matrix[2, 1] / np.sin(a2)))
    elif (Matrix[0, 1] != 0) & (Matrix[2, 1] == 0):
        # Case where cos(ay) = 0 so ay is -90° or 90°
        # Ratio of FDenominator to cos(ax) is sin(ay)
        a1 = np.real(np.arcsin(Matrix[0, 1] / np.sin(a2)))

    # Retrieve the third angle in the sequence by its tangent
    if (Matrix[1, 0] != 0) & (Matrix[1, 2] != 0):
        # % General case, tan(az) = TNumerator / TDenominator
        a3 = np.arctan2(-Matrix[1, 0], Matrix[1, 2])
        # a3 = np.real(np.arccos(Matrix[1, 2] / -np.sin(a2)))
    elif (Matrix[1, 0] == 0) & (Matrix[1, 2] == 0):
        # % Case where cos(ax) = 0, and the other two angles are not defined
        a3 = 0
    elif (Matrix[1, 2] == 0) & (Matrix[1, 2] != 0):
        # Case where sin(az) = 0 so az is 0° or 180°
        # Ratio of TNumerator to cos(ax) is cos(az)
        a3 = np.real(np.arccos(Matrix[1, 2] / -np.sin(a2)))
    elif (Matrix[1, 0] != 0) & (Matrix[1, 2] == 0):
        # Case where cos(az) = 0 so az is -90° or 90°
        # Ratio of TDenominator to cos(ax) is sin(az)
        a3 = np.real(np.arcsin(Matrix[1, 0] / np.sin(a2)))

    OutputMatrix = [a1, a2, a3]
    return OutputMatrix


def interface_numpy_to_biorbd_rotation_mat(R: np.ndarray) -> biorbd.Rotation:
    """
    This function returns the rotation matrix in biorbd formalism

    Parameters
    ---------
    R : np.ndarray
        Rotation matrix (3x3)

    Returns
    ---------
    biorbd.Rotation
        The rotation matrix object
    """

    return biorbd.Rotation(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )


def rot_mat_to_euler(rotation_matrix: np.ndarray, seq: str = "xyz"):
    """
    This function returns the rotation matrix in euler angles vector

    Parameters
    ---------
    rotation_matrix : np.ndarray
        Rotation matrix (3x3)
    seq: str = "xyz"
        order of the coordinates in the returned vector
    Returns
    ---------
    biorbd.Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()
        The Euler vector in radiant as an array
    """

    rotation_matrix_biorbd = interface_numpy_to_biorbd_rotation_mat(rotation_matrix)
    return biorbd.Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()


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
data_Clara = get_exp_data_from_pickle("q_exp_Clara.pkl")
data_file_1 = get_created_data_from_pickle(
    "converged/UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_lever_bras_no_offset_hum_low_slack_taumin25.pkl")
data_file_2 = get_created_data_from_pickle(
    "UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_15_lever_bras_no_offset_no_slack_all_muscles_derivative_true_200000_1500.pkl")
data_file_3 = get_created_data_from_pickle(
    "UpperLimbMuscle_2_lig_200_index_1_10_tau_low05_1.5_and_15_lever_bras_no_offset_no_slack_all_muscles_derivative_true_150000_1500.pkl")
data_file_4 = get_created_data_from_pickle(
    "UpperLimbMuscle_2_lig_200_range_2_10_tau_low05_1.5_and_150_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000.pkl"
)

model = biorbd.Model(
    "/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/models/wu_converted_definitif_without_floating_base.bioMod")

segment_list = [8, 15, 22]  # in order : clavicle, scapula, humerus

hum_elevation_Clara, angles_Clara = from_q_to_shoulder_euler_angles(data_Clara)
hum_elevation_file1, angles_file1 = from_q_to_shoulder_euler_angles(data_file_1)
hum_elevation_file2, angles_file2 = from_q_to_shoulder_euler_angles(data_file_2)
hum_elevation_file3, angles_file3 = from_q_to_shoulder_euler_angles(data_file_3)
hum_elevation_file4, angles_file4 = from_q_to_shoulder_euler_angles(data_file_4)
Matsurama_datas = np.zeros((3, 11))  # in order :STProtracElev, STMedRot, STTiltlevation
Matsurama_abscissa = np.zeros((3, 11))
Matsurama_datas[1, :] = [-4.64, -1.6, 1.28, 4.48, 7.84, 12.32, 17.44, 22.72, 28.16, 32.64, 37.28]
Matsurama_abscissa[1, :] = [29.954, 39.817, 49.863, 59.726, 69.772, 79.635, 90.228, 99.909, 110.137, 120, 129.863]
Matsurama_datas[0, :] = [-23.068, -22.814, -22.56, -22.305, -21.572, -20.479, -19.625, -19.731, -20.075, -21.018, -20.644]
Matsurama_abscissa[0, :] = [30.251, 40.091, 49.932, 59.954, 70.159, 80, 90.205, 99.681, 109.886, 119.909, 129.932]
Matsurama_datas[2, :] = [-7.375, -6.273, -5.271, -4.269, -2.966, -1.663, -0.06, 2.244, 4.649, 9.054, 9.86]
Matsurama_abscissa[2, :] = [29.954, 40, 49.58, 59.909, 69.954, 80, 90.046, 100.091, 110.137, 119.817, 129.863]


figX, axs = plt.subplots(4, 3)

add_datas_to_plot(hum_elevation_Clara, angles_Clara, 'b', 'Clara', axs)
add_datas_to_plot(hum_elevation_file1, angles_file1, 'r', 'file 1', axs)
add_datas_to_plot(hum_elevation_file2, angles_file2, 'g', 'file 2', axs)
add_datas_to_plot(hum_elevation_file3, angles_file3, 'k', 'file 3', axs)
add_datas_to_plot(hum_elevation_file4, angles_file4, 'm', 'best file', axs)

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
