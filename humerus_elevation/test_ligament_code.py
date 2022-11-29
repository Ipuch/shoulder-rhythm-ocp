from bioptim import OdeSolver, CostType, RigidBodyDynamics, Solver, DefectType, Shooting, SolutionIntegrator, Solution, DynamicsFcn, PlotType, BiorbdInterface, PenaltyNodeList
import biorbd_casadi as biorbd
from utils.upper_limb import UpperLimbOCP, Models
import numpy as np
import matplotlib.pyplot as plt
import pickle
from casadi import MX, norm_2


def custom_plot(nlp, x, first_marker: str, second_marker: str):

    model = nlp.model
    marker_func = biorbd.to_casadi_func("markerfunc", nlp.model.markers, nlp.states["q"].mx)

    marker_0 = biorbd.marker_index(nlp.model, first_marker)
    marker_1 = biorbd.marker_index(nlp.model, second_marker)

    q = nlp.states["q"].mapping.to_second.map(x[nlp.states["q"].index, :])

    for i in range(0, q.shape[1]):
        markersend = marker_func(q[:, i])
        markers_diffend = np.linalg.norm(markersend[:, marker_1] - markersend[:, marker_0])[np.newaxis, np.newaxis] * 1000

        if i == 0:
            diff_dimensioned = markers_diffend
        else:
            diff_dimensioned = np.concatenate((diff_dimensioned, markers_diffend), axis=1)

    return diff_dimensioned


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


def main():

    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    ode_solver = OdeSolver.COLLOCATION()

    n_threads = 8
    model_path = Models.WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLES.value
    file_name = "UpperLimbMuscle_2_lig_200_index_1_0.5_tau_low02_15_and_1500_lever_bras_no_offset_no_slack_all_muscles_derivative_true_1500_1000_qdot_50_new_obj_0.7.pkl"
    # --- Solve the program --- #
    myocp = UpperLimbOCP(
        biorbd_model_path="/home/mickaelbegon/Documents/stage_nicolas/bioptim_exo/models/wu_converted_definitif_without_floating_base.bioMod",
        n_shooting=50,
        ode_solver=ode_solver,
        rigidbody_dynamics=RigidBodyDynamics.ODE,
        n_threads=n_threads,
        # seed=42,
        seed=None,
        dynamics=DynamicsFcn.MUSCLE_DRIVEN
    )
    myocp.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    myocp.ocp.add_plot(
        "ligament_conoide_distance",
        lambda t, x, u, p: custom_plot(myocp.ocp.nlp[0], x, "CLAV_Conoid", "SCAP_Conoid"),
        # plot_type=PlotType.INTEGRATED,
    )
    myocp.ocp.add_plot(
        "ligament_trpz_distance",
        lambda t, x, u, p: custom_plot(myocp.ocp.nlp[0], x, "CLAV_TRPZ", "SCAP_TRPZ"),
        # plot_type=PlotType.PLOT,
    )

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(3000)
    # solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = myocp.ocp.solve(solv)

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    print(sol.status)
    sol.print_cost()

    out = sol.integrate(
        shooting_type=Shooting.SINGLE,
        keep_intermediate_points=False,
        merge_phases=True,
        integrator=SolutionIntegrator.SCIPY_DOP853,
    )

    plt.figure()
    plt.plot(sol.time, sol.states["q"].T, label="ocp", marker=".")
    plt.plot(out.time, out.states["q"].T, label="integrated", marker="+")
    plt.legend()
    plt.show()
    save_results(sol, file_name)
    sol.animate(show_floor=False, show_gravity=False)


if __name__ == "__main__":
    main()
