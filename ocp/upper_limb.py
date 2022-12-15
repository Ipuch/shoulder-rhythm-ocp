"""
This script is the script where the ocp used in main is constructed. The objectives, constraints and bounds are defined
here

"""

import numpy as np
from casadi import MX, norm_2, if_else
from typing import Union

import biorbd_casadi as biorbd
from bioptim import (
    OdeSolver,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    ControlType,
    InterpolationType,
    PenaltyNodeList,
    QAndQDotBounds,
    InitialGuess,
    Dynamics,
    BiorbdInterface,
    Node,
)

from ocp.utils import axemobile_yxy, get_target, euler_to_matrix_interface


class UpperLimbOCP:
    """
    Class to generate the OCP for the upper limb motion

    Methods
    ----------
    _set_dynamics
        Set the dynamics of the OCP
    _set_objective
        Set the objective of the OCP
    _set_constraints
        Set the constraints of the OCP
    _set_bounds
        method to set the bounds of the OCP
    _set_initial_guess
        method to set the initial guess of the OCP
    _set_mapping
        method to set the mapping between variables of the model
    _print_bounds
        method to print the bounds of the states into the console
    """

    def __init__(
            self,
            biorbd_model_path: str = None,
            n_shooting: int = 50,
            phase_durations: float = 0.5,
            n_threads: int = 8,
            ode_solver: OdeSolver = OdeSolver.RK4(),
            use_sx: bool = False,
            dynamics_fcn: DynamicsFcn = DynamicsFcn.MUSCLE_DRIVEN,
            slack: Union[int, float] = 0.2,
            hum_starting_orientation: list[int, float] = None,
            hum_ending_orientation: list[int, float] = None,
    ):
        """
        Parameters
        ----------
        biorbd_model_path : str
            path to the biorbd model
        n_shooting : tuple
            number of shooting points for each phase
        phase_durations : tuple
            duration of each phase
        n_threads : int
            number of threads to use for the solver
        ode_solver : OdeSolver
            type of ordinary differential equation solver to use
        use_sx : bool
            use SX for the dynamics
        slack : Union[int, float]
            slack for the bounds
        hum_starting_orientation: list[int, float]
            starting angle of the humerus in the thorax CS
        hum_ending_orientation: list[int, float]
            ending angle of the humerus in the thorax CS
        """
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting

        self.phase_durations = phase_durations
        self.phase_time = phase_durations

        self.duration = np.sum(self.phase_durations)

        self.n_threads = n_threads
        self.ode_solver = ode_solver

        if biorbd_model_path is not None:
            self.biorbd_model = biorbd.Model(biorbd_model_path)
            self.marker_labels = [m.to_string() for m in self.biorbd_model.markerNames()]

            self.n_q = self.biorbd_model.nbQ()
            self.n_qdot = self.biorbd_model.nbQdot()
            self.nb_root = self.biorbd_model.nbRoot()

            self.n_tau = self.biorbd_model.nbGeneralizedTorque() - self.biorbd_model.nbRoot()
            self.n_mus = self.biorbd_model.nbMuscleTotal()

            self.tau_min, self.tau_init, self.tau_max = -25, 0, 25
            self.muscle_min, self.muscle_max, self.muscle_init = 0, 1, 0.10
            self.humerus_starting_elevation = euler_to_matrix_interface(hum_starting_orientation)
            self.humerus_ending_elevation = euler_to_matrix_interface(hum_ending_orientation)


            self.dynamics_fcn = dynamics_fcn
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()

            self.slack = slack
            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()

            self.initial_states = []

            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()

            self._set_init_ref_for_bounds()
            # reload it to get the new thorax values
            self.biorbd_model = biorbd.Model(biorbd_model_path)

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self._set_dynamics()
            self._set_objective_functions()
            self._set_constraint_functions()

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                n_shooting=self.n_shooting,
                phase_time=self.phase_durations,
                x_init=self.x_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                n_threads=n_threads,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """

        if self.dynamics_fcn == DynamicsFcn.MUSCLE_DRIVEN:
            self.dynamics = Dynamics(self.dynamics_fcn, with_torque=True)
        else:
            self.dynamics = Dynamics(self.dynamics_fcn)

    def _set_objective_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

        def custom_ligaments_distance(all_pn: PenaltyNodeList, first_marker: str, second_marker: str, L_ref: float) -> MX:
            """
            Calculates the distance between two markers and maxing it to a value given by L_ref

            Parameters
            ----------
            first_marker: str
                Name of the first marker
            second_marker: str
                Name of the second marker
            L_ref: float
                Distance of reference the ligament can not excess

            Returns
            ----------
                Distance between the two markers
                """

            marker_0 = biorbd.marker_index(all_pn.nlp.model, first_marker)
            marker_1 = biorbd.marker_index(all_pn.nlp.model, second_marker)

            markers = all_pn.nlp.model.markers(all_pn.nlp.states["q"].mx)
            markers_diff = markers[marker_1].to_mx() - markers[marker_0].to_mx()
            markers_diff = BiorbdInterface.mx_to_cx("markers", markers_diff, all_pn.nlp.states["q"])

            markers_diff_norm = norm_2(markers_diff)

            return if_else(
                markers_diff_norm > L_ref,
                markers_diff_norm-L_ref,
                0
            )

        # MINIMIZE Q
        # Reduces the movements of the scapula and the clavicle, in order to keep a physiological look-alike movement
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=1, weight=10)
        # tried :
        # range2 (doesn't converge)
        # range 5(converges but bad motion)
        # with constraint, works with range 5, weight of 0.1

        # MINIMIZE QDOT
        # restricts the speed of the movements to be realistic and have a full frame movement
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=50)
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True, weight=0.5
        )  # added

        # MINIMIZE MUSCLE ACTIVATIONS
        if self.dynamics.type == DynamicsFcn.MUSCLE_DRIVEN:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True, weight=1500)

        # MINIMIZE TAU
        # Minimize the tau separately on the first 5 DDL and the rest. Play with these to correct the scapula's sag
        # and at the same time have realistic values of tau.
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(5, self.n_tau), weight=15  # tried range2, doesn't converge
        )
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(5), weight=1.5)

        # MINIMIZE TAU DERIVATIVE TRUE
        # restricts the changes from one moment to the next one to have better continuity
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500
        )

        # MINIMIZE QDOT MAYER
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=100, node=Node.END)

        # CUSTOM OBJ
        # custom objectives to simulate the 2 claviculo-scapular ligaments

        target_conoid, target_trpz = get_target(self.biorbd_model_path, self.n_shooting + 1)
        self.objective_functions.add(
            custom_ligaments_distance,
            first_marker='CLAV_Conoid',
            second_marker='SCAP_Conoid',
            custom_type=ObjectiveFcn.Lagrange,
            quadratic=True,
            node=Node.ALL,
            L_ref=target_conoid[0],
            target=target_conoid,
            weight=200,
        )
        self.objective_functions.add(
            custom_ligaments_distance,
            first_marker='CLAV_TRPZ',
            second_marker='SCAP_TRPZ',
            custom_type=ObjectiveFcn.Lagrange,
            quadratic=True,
            node=Node.ALL,
            L_ref=target_trpz[0],
            weight=200,
        )

        def custom_func_track_position_in_GCS(all_pn: PenaltyNodeList, humerus_segment: str, thorax_segment: str) -> MX:
            """
            Gives the angles between the humerus and the thorax. this lets us control the elevation of the humerus
            by only giving the value of thoraco-humeral elevation.
            It also lets us give the plane of elevation by changing the rotation index
            the thoraco-humeral elevation follows the yxy sequence with in order :
            - Plane of elevation, 0 degrees is abduction, 90 degrees is forward flexion. (index = 0)
            - elevation (index = 1)
            - axial rotation, endo- or internal rotation and exo- or external-rotation (index = 2)

            Parameters
            ----------
            humerus_segment: str
                Name of the humerus segment in the bioMod
            thorax_segment: str
                Name of the thorax segment in the bioMod

            Returns
            ----------
                Each values of the rotation matrix
            """
            model = all_pn.nlp.model
            rotation_matrix_segment_index = biorbd.segment_index(model, humerus_segment)
            rotation_matrix_thorax_index = biorbd.segment_index(model, thorax_segment)
            q = all_pn.nlp.states["q"].mx
            # global JCS gives the local matrix according to the global matrix
            segment_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_segment_index).rot().to_mx()
            thorax_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_thorax_index).rot().to_mx()

            rot_matrix_hum_in_thoraxJCS = thorax_mx_matrix.T @ segment_mx_matrix

            b = MX.zeros(9)
            b[:] = rot_matrix_hum_in_thoraxJCS[:9]
            output_casadi = BiorbdInterface.mx_to_cx("scal_prod", b, all_pn.nlp.states["q"])

            return output_casadi

        # These two objectives are here to control the starting and ending point of the elevation
        self.objective_functions.add(
            custom_func_track_position_in_GCS,
            target=self.humerus_starting_elevation,
            humerus_segment="humerus",
            thorax_segment="thorax",
            custom_type=ObjectiveFcn.Mayer,
            quadratic=True,
            node=Node.START,
            weight=20000,
            )
        self.objective_functions.add(
            custom_func_track_position_in_GCS,
            target=self.humerus_ending_elevation,
            humerus_segment="humerus",
            thorax_segment="thorax",
            custom_type=ObjectiveFcn.Mayer,
            quadratic=True,
            node=Node.END,
            weight=20000,
        )

    def _set_constraint_functions(self):
        """
        Set the multi-constraint functions for each phase with specific weights
        """

        def custom_func_track_position_in_GCS(
          all_pn: PenaltyNodeList,
          humerus_segment: str,
          thorax_segment: str,
          ) -> MX:
            """
            Gives the angles between the humerus and the thorax. this lets us control the elevation of the humerus
            by only giving the value of thoraco-humeral elevation.
            It also lets us give the plane of elevation by changing the rotation index
            the thoraco-humeral elevation follows the yxy sequence with in order :
            - Plane of elevation, 0 degrees is abduction, 90 degrees is forward flexion. (rot_index = 0)
            - elevation (rot_index = 1)
            - axial rotation, endo- or internal rotation and exo- or external-rotation (rot_index = 2)

            Parameters
            ----------
            humerus_segment: str
                Name of the humerus segment in the bioMod
            thorax_segment: str
                Name of the thorax segment in the bioMod
            rot_index: int
                Distance of reference the ligament can not excess

            Returns
            ----------
                Distance between the two markers
            """
            model = all_pn.nlp.model
            rotation_matrix_segment_index = biorbd.segment_index(model, humerus_segment)
            rotation_matrix_thorax_index = biorbd.segment_index(model, thorax_segment)
            q = all_pn.nlp.states["q"].mx
            # global JCS gives the local matrix according to the global matrix
            segment_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_segment_index).rot().to_mx()
            thorax_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_thorax_index).rot().to_mx()

            rot_matrix_hum_in_thoraxJCS = thorax_mx_matrix.T @ segment_mx_matrix

            b = MX.zeros(9)
            b[:] = rot_matrix_hum_in_thoraxJCS[:9]
            output_casadi = BiorbdInterface.mx_to_cx("scal_prod", b, all_pn.nlp.states["q"])

            return output_casadi

        # These two constraints are here to control the starting and ending point of the elevation
        # self.constraints.add(
        #     custom_func_track_position_in_GCS,
        #     target=self.humerus_starting_elevation,
        #     humerus_segment="humerus",
        #     thorax_segment="thorax",
        #     quadratic=True,
        #     node=Node.START,
        #     weight=2000,
        #     )
        # self.constraints.add(
        #     custom_func_track_position_in_GCS,
        #     target=self.humerus_ending_elevation,
        #     humerus_segment="humerus",
        #     thorax_segment="thorax",
        #     quadratic=True,
        #     node=Node.END,
        #     weight=2000,
        # )

    def _set_initial_guesses(self):
        """
        Set the initial guess for the optimal control problem (states and controls)
        """
        # --- Initial guess --- #

        # linear interpolation between initial and final states of x_init_ref[0, :] and x_init_ref[-1, :]
        x_init_linear = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1))
        for i in range(self.n_q):
            x_init_linear[i, :] = np.linspace(self.x_init_ref[0, i], self.x_init_ref[-1, i], self.n_shooting + 1)

        self.x_init = InitialGuess(x_init_linear, interpolation=InterpolationType.EACH_FRAME)
        if self.dynamics_fcn == DynamicsFcn.MUSCLE_DRIVEN:
            self.u_init = InitialGuess(
                [self.tau_init] * self.n_tau + [self.muscle_init] * self.biorbd_model.nbMuscles())
        else:
            self.u_init = InitialGuess(self.u_init_ref, interpolation=InterpolationType.EACH_FRAME)

    def _set_init_ref_for_bounds(self):
        """
        function to set the x_init_ref and u_init_ref that are used in the bounds and initial guess

        """
        q_ref_homemade = np.zeros((self.biorbd_model.nbQ(), 2))
        q_ref_start = np.zeros((self.biorbd_model.nbQ()))
        q_ref_end = np.zeros((self.biorbd_model.nbQ()))

        # frontal plane
        q_ref_start[0] = -0.06  # clavicle retraction
        q_ref_start[1] = -0.26  # clavicle elevation
        q_ref_start[2] = -0.03  # scapula y rotation
        q_ref_start[3] = 0.19  # scapula x rotation
        q_ref_start[4] = 0.27  # scapula z rotation
        q_ref_start[5] = 0.05  # 0.05 for frontal, 0.20 for scap plane, 1.35/40 for sagittal + other mods
        q_ref_start[6] = 0.30  # hum elevation
        q_ref_start[7] = 0  # humerus intern rotation

        q_ref_end[0] = -0.20  # clavicle retraction
        q_ref_end[1] = 0.10  # clavicle elevation
        q_ref_end[2] = 0.52  # scapula y rotation
        q_ref_end[3] = 0.12  # scapula x rotation
        q_ref_end[4] = 0.28  # scapula z rotation
        q_ref_end[5] = -0.09  # 0.05 for frontal, 0.20 for scap plane, 1.35/40 for sagittal + other mods
        q_ref_end[6] = 0.7  # hum elevation
        q_ref_end[7] = 0  # humerus intern rotation

        q_ref_homemade = np.concatenate([[q_ref_start], [q_ref_end]], axis=0)

        tau_ref_homemade = np.zeros((10, 50))
        qdot_ref_homemade = np.zeros((2, 10))
        # building initial guess
        self.x_init_ref = np.concatenate([q_ref_homemade, qdot_ref_homemade], axis=1)  # without floating base
        self.u_init_ref = tau_ref_homemade  # used in the initial guess

    def _set_boundary_conditions(self):
        """
        Set the boundary conditions for controls and states for each phase.
        """
        self.x_bounds = QAndQDotBounds(self.biorbd_model)
        x_slack_start = 0
        # x_slack_start = 0.1 * np.ones(self.x_init_ref[: self.n_q, 0].shape)
        self.x_bounds.min[: self.n_q, 0] = self.x_init_ref[0, : self.n_q]
        self.x_bounds.max[: self.n_q, 0] = self.x_init_ref[0, : self.n_q]

        x_slack_end = self.slack * np.ones(self.x_init_ref[-1, : self.n_q].shape)
        self.x_bounds.min[: self.n_q, -1] = self.x_init_ref[-1, : self.n_q] - x_slack_end
        self.x_bounds.max[: self.n_q, -1] = self.x_init_ref[-1, : self.n_q] + x_slack_end

        self.x_bounds.min[self.n_q:, 0] = [-1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, 0] = [1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.min[self.n_q:, -1] = [-5e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, -1] = [5e-3] * self.biorbd_model.nbQdot()

        if self.dynamics_fcn == DynamicsFcn.MUSCLE_DRIVEN:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau + [self.muscle_min] * self.biorbd_model.nbMuscleTotal(),
                [self.tau_max] * self.n_tau + [self.muscle_max] * self.biorbd_model.nbMuscleTotal(),
            )
        else:
            self.u_bounds.add(
                [self.tau_min] * self.n_tau,
                [self.tau_max] * self.n_tau,
            )
        self.u_bounds[0][5:8] = 0
        self.u_bounds[0][5:8] = 0
