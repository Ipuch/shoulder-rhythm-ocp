from typing import Union

import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, norm_2, if_else, acos, dot, arccos, arctan2

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


def axemobile_yxy(matrix: Union[np.ndarray, MX]):
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

    OutputMatrix = np.zeros((3, 3)) if not matrix.type_name() == 'MX' else MX.zeros((3, 3))
    Matrix = matrix

    a1 = 0 if not matrix.type_name() == 'MX' else MX.zeros(1)
    a2 = np.arccos(Matrix[1, 1]) if not matrix.type_name() == 'MX' else arccos(Matrix[1, 1]) # Retrieve the second angle in the sequence by its sinus
    a3 = 0 if not matrix.type_name() == 'MX' else MX.zeros(1)

    if matrix.type_name() == 'MX':
        OutputMatrix[0] = arctan2(Matrix[0, 1], Matrix[2, 1])
        OutputMatrix[1] = arccos(Matrix[1, 1])
        OutputMatrix[2] = arctan2(Matrix[1, 0], -Matrix[1, 2])
    else:
        # Retrieve the first angle in the sequence by its tangent
        if (Matrix[0, 1] != 0) & (Matrix[2, 1] != 0):
            # General case, tan(ay) = FNumerator / FDenominator
            a1 = np.arctan2(Matrix[0, 1], Matrix[2, 1]) if not matrix.type_name() == 'MX' else arctan2(Matrix[0, 1], Matrix[2, 1])
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