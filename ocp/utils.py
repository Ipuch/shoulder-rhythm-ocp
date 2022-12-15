import numpy as np
from typing import Union
try:
    import biorbd
except:
    import biorbd_casadi as biorbd
from casadi import MX, arccos, arctan2
from bioptim import Solution
import pickle


def get_starting_conoid_trpz_length(model_path: str = None):
    """
    Get the norm from both Conoid ligament and trapezoidal ligament

    parameters
    ----------
    model_path: str
        the path to the model
    """
    biorbd_model_path = model_path
    biorbd_model = biorbd.Model(biorbd_model_path)

    cs_clav_con = biorbd_model.anatomicalMarkers(np.zeros(biorbd_model.nbQ()))[0].to_array()
    cs_scap_con = biorbd_model.anatomicalMarkers(np.zeros(10))[2].to_array()
    con_ligament = cs_scap_con - cs_clav_con
    norm_con_ligament = np.linalg.norm(con_ligament)

    cs_clav_trpz = biorbd_model.anatomicalMarkers(np.zeros(10))[1].to_array()
    cs_scap_trpz = biorbd_model.anatomicalMarkers(np.zeros(10))[3].to_array()
    trpz_ligament = cs_scap_trpz - cs_clav_trpz
    norm_trpz_ligament = np.linalg.norm(trpz_ligament)

    return norm_con_ligament, norm_trpz_ligament


def axemobile_yxy(matrix: np.ndarray | MX) -> list[np.ndarray, ...] | list[MX, ...]:
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

    Matrix = matrix
    OutputMatrix = np.zeros((3, 3)) if isinstance(Matrix, np.ndarray) else MX.zeros((3, 3))

    a1 = 0 if isinstance(Matrix, np.ndarray) else MX.zeros(1)
    a2 = np.arccos(Matrix[1, 1]) if isinstance(Matrix, np.ndarray) else arccos(
        Matrix[1, 1])  # Retrieve the second angle in the sequence by its sinus
    a3 = 0 if isinstance(Matrix, np.ndarray) else MX.zeros(1)

    if isinstance(Matrix, MX):
        OutputMatrix[0] = arctan2(Matrix[0, 1], Matrix[2, 1])
        OutputMatrix[1] = arccos(Matrix[1, 1])
        OutputMatrix[2] = arctan2(Matrix[1, 0], -Matrix[1, 2])
    else:
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


def interface_numpy_3x3_to_biorbd_rotation_mat(R: np.ndarray) -> biorbd.Rotation:
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


def interface_numpy_3x1_to_biorbd_rotation_mat(R: np.ndarray) -> biorbd.Rotation:
    """
    This function returns the rotation matrix in biorbd formalism

    Parameters
    ---------
    R : np.ndarray
        Rotation matrix (3x1)

    Returns
    ---------
    biorbd.Rotation
        The rotation matrix object
    """

    return biorbd.Rotation(
        R[0],
        R[1],
        R[2],
    )


def rot_mat_to_euler(rotation_matrix: np.ndarray, seq: str = "xyz") -> np.ndarray:
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

    rotation_matrix_biorbd = interface_numpy_3x3_to_biorbd_rotation_mat(rotation_matrix)
    return biorbd.Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()


def get_target(model: str, n_shooting: int):
    target_value_cor, target_value_trpz = get_starting_conoid_trpz_length(model)  # the length of both ligaments
    target_conoid = np.repeat(target_value_cor, repeats=n_shooting)
    target_trpz = np.repeat(target_value_trpz, repeats=n_shooting)

    return target_conoid, target_trpz


def save_results(sol: Solution, c3d_file_path: str, ):
    """
    Solving the ocp
    Parameters
    ----------
    sol: Solution
        The solution to the ocp at the current pool
    c3d_file_path: str
        The path to the c3d file of the task
    """

    data = dict(
        states=sol.states,
        controls=sol.controls,
        parameters=sol.parameters,
        iterations=sol.iterations,
        cost=sol.cost,
        detailed_cost=sol.detailed_cost,
        real_time_to_optimize=sol.real_time_to_optimize,
        status=sol.status
    )
    with open(f"{c3d_file_path}", "wb") as file:
        pickle.dump(data, file)


def euler_to_matrix_interface(euler_angles: list[int, float]):

    angles = np.zeros(3)
    angles[:] = euler_angles
    matrix_3x3 = biorbd.Rotation.fromEulerAngles(angles, "yxy").to_array()
    matrix_9x1 = np.zeros(9)
    matrix_9x1[0:3], matrix_9x1[3:6], matrix_9x1[6:9] = matrix_3x3[:, 0], matrix_3x3[:, 1], matrix_3x3[:, 2]

    return matrix_9x1
