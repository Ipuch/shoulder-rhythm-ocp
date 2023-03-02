"""
This script is a duplicate of the main one to use for debug
"""

from bioptim import OdeSolver, CostType, Solver, DynamicsFcn
from ocp.upper_limb import UpperLimbOCP
from viz import add_custom_plot
from utils import save_results
import numpy as np


def main():

    ode_solver = OdeSolver.COLLOCATION()

    n_threads = 8
    model_path = "../models/wu_converted_definitif_without_floating_base.bioMod"
    file_name = "../simulation_results/MuscleDriven_constraint_with_slack_test.pkl"
    # --- Solve the program --- #
    myocp = UpperLimbOCP(
        biorbd_model_path=model_path,
        n_shooting=50,
        ode_solver=ode_solver,
        n_threads=n_threads,
        dynamics_fcn=DynamicsFcn.MUSCLE_DRIVEN,
        slack=0.50,
        hum_starting_orientation=[25, 10, 0],
        hum_ending_orientation=[30, 75, 0]
    )
    myocp.ocp.add_plot_penalty(CostType.ALL)
    # humanoid.ocp.print()

    add_custom_plot(myocp, model_path)

    solv = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solv.set_maximum_iterations(3000)
    # solv.set_linear_solver("ma57")
    solv.set_print_level(5)
    sol = myocp.ocp.solve(solv)

    # --- Show results --- #
    sol.graphs(show_bounds=True)
    print(sol.status)
    sol.print_cost()

    save_results(sol, file_name)
    sol.animate(show_floor=False, show_gravity=False)


if __name__ == "__main__":
    main()
