from enum import Enum
from pathlib import Path
import numpy as np
from casadi import MX, norm_2, if_else, acos, dot

import biorbd_casadi as biorbd
from bioptim import (
    OdeSolver,
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveFcn,
    ConstraintList,
    ObjectiveList,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    ControlType,
    InterpolationType,
    RigidBodyDynamics,
    PenaltyNodeList,
    QAndQDotBounds,
    InitialGuess,
    Dynamics,
    BiorbdInterface,
    Node,
)
from sandbox.calcul_distance_ligaments import get_target_value
from models.enums import Models
from utils.Utils import axemobile_yxy, interface_numpy_to_biorbd_rotation_mat

class Tasks(Enum):
    """
    Selection of tasks
    """

    HEAD = Path(__file__).parent.parent.__str__() + "/data/F0_tete_05"


def eul2quat(eul: np.ndarray) -> np.ndarray:
    rotation_matrix = biorbd.Rotation_fromEulerAngles(eul, "xyz")
    quat = biorbd.Quaternion_fromMatrix(rotation_matrix).to_array().squeeze()
    return quat


def quat2eul(quat: np.ndarray) -> np.ndarray:
    quat_biorbd = biorbd.Quaternion(quat[0], quat[1], quat[2], quat[3])
    rotation_matrix = biorbd.Quaternion.toMatrix(quat_biorbd)
    eul = biorbd.Rotation_toEulerAngles(rotation_matrix, "xyz").to_array()
    return eul


class UpperLimbOCP:
    """
    Class to generate the OCP for the upper limb motion for a daily task of living

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
            rigidbody_dynamics: RigidBodyDynamics = RigidBodyDynamics.ODE,
            task: Tasks = Tasks.HEAD,
            use_sx: bool = False,
            seed: int = None,
            dynamics: DynamicsFcn = DynamicsFcn.MUSCLE_DRIVEN,
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
        rigidbody_dynamics : RigidBodyDynamics
            type of dynamics to use
        use_sx : bool
            use SX for the dynamics
        seed : int
            seed for the random generator
        """
        self.biorbd_model_path = biorbd_model_path
        self.n_shooting = n_shooting

        self.task = task
        self.c3d_path = f"{self.task.value}.c3d"

        self.phase_durations = phase_durations
        self.phase_time = phase_durations

        self.duration = np.sum(self.phase_durations)

        self.n_threads = n_threads
        self.ode_solver = ode_solver

        if biorbd_model_path is not None:
            self.biorbd_model = biorbd.Model(biorbd_model_path)
            self.marker_labels = [m.to_string() for m in self.biorbd_model.markerNames()]
            self.rigidbody_dynamics = rigidbody_dynamics

            self.n_q = self.biorbd_model.nbQ()
            self.n_qdot = self.biorbd_model.nbQdot()
            self.nb_root = self.biorbd_model.nbRoot()

            self.n_tau = self.biorbd_model.nbGeneralizedTorque() - self.biorbd_model.nbRoot()
            self.n_mus = self.biorbd_model.nbMuscleTotal()

            self.tau_min, self.tau_init, self.tau_max = -25, 0, 25
            self.muscle_min, self.muscle_max, self.muscle_init = 0, 1, 0.10

            self.dynamics = dynamics
            self.constraints = ConstraintList()
            self.objective_functions = ObjectiveList()

            self.x_bounds = BoundsList()
            self.u_bounds = BoundsList()

            self.initial_states = []

            self.x_init = InitialGuessList()
            self.u_init = InitialGuessList()

            self._get_experimental_data()
            # reload it to get the new thorax values
            self.biorbd_model = biorbd.Model(biorbd_model_path)

            self._set_boundary_conditions()
            self._set_initial_guesses()

            self._set_dynamics()
            self._set_objective_functions()

            if seed is not None:
                self.xn_init = InitialGuessList()
                self.un_init = InitialGuessList()

                q_noise_magnitude = np.repeat(0.005, self.n_q)
                qdot_noise_magnitude = np.repeat(0.001, self.n_qdot)
                x_noise_magnitude = np.concatenate((q_noise_magnitude, qdot_noise_magnitude))

                self.x_init.add([0] * (self.n_q + self.n_qdot), interpolation=InterpolationType.EACH_FRAME)
                self.x_init.add_noise(
                        bounds=self.x_bounds,
                        magnitude=x_noise_magnitude,
                        n_shooting=self.n_shooting + 1,
                        bound_push=0.1,
                        seed=seed,
                    )

                torque_noise_magnitude = np.repeat(0.005, self.n_tau)
                torque_noise_magnitude[5:8] = 0
                muscle_noise_magnitude = np.repeat(0.01, self.n_mus)
                u_noise_magnitude = np.concatenate((torque_noise_magnitude, muscle_noise_magnitude))

                self.u_init.add([0] * self.n_tau)
                self.un_init.add_noise(
                        bounds=self.u_bounds,
                        magnitude=u_noise_magnitude,
                        n_shooting=self.n_shooting,
                        bound_push=0,
                        seed=seed,
                )

            self.ocp = OptimalControlProgram(
                self.biorbd_model,
                self.dynamics,
                n_shooting=self.n_shooting,
                phase_time=self.phase_durations,
                x_init=self.x_init if seed is None else self.xn_init,
                x_bounds=self.x_bounds,
                u_init=self.u_init if seed is None else self.un_init,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                n_threads=n_threads,
                control_type=ControlType.CONSTANT,
                ode_solver=ode_solver,
                use_sx=use_sx,
            )

    def _get_experimental_data(self):

        model_template_path = Models.WU_WITHOUT_FLOATING_BASE_OFFSET_TEMPLATE.value
        new_biomod_file = Models.WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLES.value

        # get key events
        start_frame = 4
        end_frame = 179

        # enforced time in most cases
        phase_time = self.phase_time

        q_ref_homemade = np.zeros((self.biorbd_model.nbQ(), 51))

        # frontal plane
        q_ref_homemade[0, :] = np.linspace(-0.02, -0.21, 51)
        q_ref_homemade[1, :] = np.linspace(-0.23, 0.09, 51)
        q_ref_homemade[2, :] = np.linspace(-0.09, 0.15, 51)
        q_ref_homemade[3, :] = np.linspace(0.13, 0.04, 51)
        q_ref_homemade[4, :] = np.linspace(0.22, 0.22, 51)
        q_ref_homemade[5, :] = np.linspace(-0.11, -0.11, 51)
        q_ref_homemade[6, :] = np.linspace(0.13, 0.70, 51)  # hum elevation
        q_ref_homemade[7, :] = np.linspace(-0.82, -0.82, 51)

        tau_ref_homemade = np.zeros((10, 50))
        qdot_ref_homemade = np.zeros((10, 51))
        # building initial guess
        self.x_init_ref = np.concatenate([q_ref_homemade, qdot_ref_homemade])  # without floating base
        self.u_init_ref = tau_ref_homemade

    def _set_dynamics(self):
        """
        Set the dynamics of the optimal control problem
        """

        if self.rigidbody_dynamics == RigidBodyDynamics.ODE:
            if self.dynamics == DynamicsFcn.MUSCLE_DRIVEN:
                self.dynamics = Dynamics(self.dynamics, with_torque=True)
            else:
                self.dynamics = Dynamics(self.dynamics)
        else:
            raise ValueError("This dynamics has not been implemented")

    def _set_objective_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

        def custom_ligaments_distance(all_pn: PenaltyNodeList, first_marker: str, second_marker: str, L_ref: float) -> MX:

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

        def custom_func_track_position_in_GCS(all_pn: PenaltyNodeList, segment: str, thorax_segment: str) -> MX:
            model = all_pn.nlp.model
            rotation_matrix_segment_index = biorbd.segment_index(model, segment)
            rotation_matrix_thorax_index = biorbd.segment_index(model, thorax_segment)
            q = all_pn.nlp.states["q"].mx
            # global JCS gives the local matrix according to the global matrix
            segment_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_segment_index).to_mx()
            thorax_mx_matrix = all_pn.nlp.model.globalJCS(q, rotation_matrix_thorax_index).to_mx()

            rot_matrix_hum_in_thoraxJCS = thorax_mx_matrix.T @ segment_mx_matrix

            b = axemobile_yxy(rot_matrix_hum_in_thoraxJCS)
            output_casadi = BiorbdInterface.mx_to_cx("scal_prod", b[1], all_pn.nlp.states["q"])

            return output_casadi

        target_value_cor, target_value_trpz = get_target_value(self.biorbd_model_path)  # the length of both ligaments
        target_conoid = np.repeat(target_value_cor, repeats=self.n_shooting + 1)
        target_trpz = np.repeat(target_value_trpz, repeats=self.n_shooting + 1)

        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=1, weight=0.5)  # tried range2 (doesn't converge), range 5(converges but bad motion)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=50)
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", derivative=True, weight=0.5
        )  # added
        if self.dynamics.type == DynamicsFcn.MUSCLE_DRIVEN:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1000)
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True, weight=1500)
        # self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=15)
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(2, self.n_tau), weight=1500  # tried range2, doesn't converge
        )
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(2), weight=15)
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1500
        )
        self.objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=100, node=Node.END)
        self.objective_functions.add(
            custom_func_track_position_in_GCS,
            target=0.7,
            segment="humerus",
            thorax_segment="thorax",
            custom_type=ObjectiveFcn.Mayer,
            quadratic=True,
            node=Node.END,
            weight=10000
            )
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
            # target=target_trpz,
            weight=200,
        )

    def _set_constraint_functions(self):
        """
        Set the multi-objective functions for each phase with specific weights
        """

    def _set_initial_guesses(self):
        """
        Set the initial guess for the optimal control problem (states and controls)
        """
        # --- Initial guess --- #

        # linear interpolation between initial and final states of x_init_ref[0, :] and x_init_ref[-1, :]
        x_init_linear = np.zeros((self.n_q + self.n_qdot, self.n_shooting + 1))
        for i in range(self.n_q):
            x_init_linear[i, :] = np.linspace(self.x_init_ref[i, 0], self.x_init_ref[i, -1], self.n_shooting + 1)

        self.x_init = InitialGuess(x_init_linear, interpolation=InterpolationType.EACH_FRAME)
        if self.dynamics == DynamicsFcn.MUSCLE_DRIVEN:
            self.u_init = InitialGuess(
                [self.tau_init] * self.n_tau + [self.muscle_init] * self.biorbd_model.nbMuscles())
        else:
            self.u_init = InitialGuess(self.u_init_ref, interpolation=InterpolationType.EACH_FRAME)

    def _set_boundary_conditions(self):
        """
        Set the boundary conditions for controls and states for each phase.
        """
        self.x_bounds = QAndQDotBounds(self.biorbd_model)
        x_slack_start = 0
        # x_slack_start = 0.1 * np.ones(self.x_init_ref[: self.n_q, 0].shape)
        self.x_bounds.min[: self.n_q, 0] = self.x_init_ref[: self.n_q, 0]
        self.x_bounds.max[: self.n_q, 0] = self.x_init_ref[: self.n_q, 0]

        x_slack_end = 0.15 * np.ones(self.x_init_ref[: self.n_q, -1].shape)
        self.x_bounds.min[: self.n_q, -1] = self.x_init_ref[: self.n_q, -1] - x_slack_end
        self.x_bounds.max[: self.n_q, -1] = self.x_init_ref[: self.n_q, -1] + x_slack_end

        # norm of the quaternion should be 1 at the start and at the end
        if self.biorbd_model.nbQuat() > 0:
            self.x_bounds.min[5:8, 0] = self.x_init_ref[5:8, 0]
            self.x_bounds.max[5:8, 0] = self.x_init_ref[5:8, 0]
            self.x_bounds.min[5:8, -1] = self.x_init_ref[5:8, -1]
            self.x_bounds.max[5:8, -1] = self.x_init_ref[5:8, -1]

            self.x_bounds.min[10, 0] = self.x_init_ref[10, 0]
            self.x_bounds.max[10, 0] = self.x_init_ref[10, 0]
            self.x_bounds.min[10, -1] = self.x_init_ref[10, -1]
            self.x_bounds.max[10, -1] = self.x_init_ref[10, -1]

        self.x_bounds.min[self.n_q:, 0] = [-1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, 0] = [1e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.min[self.n_q:, -1] = [-5e-3] * self.biorbd_model.nbQdot()
        self.x_bounds.max[self.n_q:, -1] = [5e-3] * self.biorbd_model.nbQdot()

        if self.biorbd_model.nbQuat() > 0:
            self.x_bounds.min[8:10, 1], self.x_bounds.min[10, 1] = self.x_bounds.min[9:11, 1], -1
            self.x_bounds.max[8:10, 1], self.x_bounds.max[10, 1] = self.x_bounds.max[9:11, 1], 1

        if self.dynamics == DynamicsFcn.MUSCLE_DRIVEN:
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
