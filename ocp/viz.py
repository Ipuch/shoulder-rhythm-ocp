import biorbd
import biorbd_casadi as biorbd_cas
from ocp.upper_limb import UpperLimbOCP
import numpy as np
from ocp.utils import axemobile_yxy


def custom_plot_markers_distance(nlp, x, first_marker: str, second_marker: str):
    """
    plots the distance between the two chosen markers

    Parameters
    ----------
    first_marker : str
        Name of the first marker
    second_marker : str
        Name of the second marker

    Returns
    -------
    The distance between the two markers, in millimeters
    """
    model = nlp.model
    marker_func = biorbd_cas.to_casadi_func("markerfunc", nlp.model.markers, nlp.states["q"].mx)

    marker_0 = biorbd_cas.marker_index(model, first_marker)
    marker_1 = biorbd_cas.marker_index(model, second_marker)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])

    for i in range(0, q.shape[1]):
        markersend = marker_func(q[:, i])
        markers_diffend = np.linalg.norm(markersend[:, marker_1] - markersend[:, marker_0])[np.newaxis, np.newaxis] * 1000

        if i == 0:
            diff_dimensioned = markers_diffend
        else:
            diff_dimensioned = np.concatenate((diff_dimensioned, markers_diffend), axis=1)

    return diff_dimensioned


def custom_plot_thorax_hum_angle(nlp, x, thorax_segment: str, humerus_segment: str, model: str):
    """
    adds plots custom plots for the ocp

    Parameters
    ----------
    thorax_segment : str
        Name of the first marker
    humerus_segment : str
        Name of the second marker
    model : str
        Name of the model

    Returns
    -------
    The angle between the humerus and the thorax, following the ISB recommendations
    """

    biorbd_model = biorbd.Model(model)

    seg_thorax = biorbd.segment_index(nlp.model, thorax_segment)
    seg_humerus = biorbd.segment_index(nlp.model, humerus_segment)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])
    diff_dimensioned = np.zeros((1, 1))
    if q.shape[1] == 2:
        theta_dimensioned = np.zeros((1, 2))
        return theta_dimensioned
    else:
        for i in range(q.shape[1]):
            thorax_position = biorbd_model.globalJCS(q[:, i], seg_thorax).rot().to_array()
            hum_position = biorbd_model.globalJCS(q[:, i], seg_humerus).rot().to_array()

            hum1_thorax_rotation = thorax_position.T @ hum_position  # rotation matrix of TH

            rot_hum_euler = axemobile_yxy(hum1_thorax_rotation)

            theta_euler = rot_hum_euler[1][np.newaxis, np.newaxis]

            if i == 0:
                diff_dimensioned = theta_euler
            else:
                diff_dimensioned = np.concatenate((diff_dimensioned, theta_euler), axis=1)

        return diff_dimensioned


def add_custom_plot(myocp: UpperLimbOCP, model_path: str):
    """
    Parameters
    ----------
    myocp : UpperLimbOCP
       the ocp to do the plots
    model_path : str
        the path of the model

    Returns
    -------
    The plots of the length of the conoid and trapezoid ligaments and the angle of the humerus in the thorax CS
    """

    myocp.ocp.add_plot(
        "ligament_conoide_distance",
        lambda t, x, u, p: custom_plot_markers_distance(myocp.ocp.nlp[0], x, "CLAV_Conoid", "SCAP_Conoid"),
        # plot_type=PlotType.INTEGRATED,
    )
    myocp.ocp.add_plot(
        "ligament_trpz_distance",
        lambda t, x, u, p: custom_plot_markers_distance(myocp.ocp.nlp[0], x, "CLAV_TRPZ", "SCAP_TRPZ"),
        # plot_type=PlotType.PLOT,
    )
    myocp.ocp.add_plot(
        "Humerus position relative to the Thorax",
        lambda t, x, u, p: custom_plot_thorax_hum_angle(myocp.ocp.nlp[0], x, "thorax", "humerus", model_path),
        # plot_type=PlotType.PLOT,
    )
