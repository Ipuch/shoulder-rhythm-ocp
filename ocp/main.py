"""
This scripts does an elevation of the humerus in a chosen plane. Default plane is scapular plane.
To change it, change the 'hum_plane_angle' value.

The results are saved in the datas repertory. You can choose the name of the file by changing 'file_name'.

You can choose the starting and ending angles of the humerus elevation. Default values are 0.3 to 0.8 radiant
To change it, change the 'hum_start_elevation' and 'hum_final_elevation' values.

You can also change the slacks of the bounds by changing 'slack'. Default value is 0.2

The model used in this script is the "wu_converted_definitif_without_floating_base", with segments following the ISB
recommendations

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
    file_name = "../datas/UpperLimbMuscle_default_constraint_test_tau_q_test_slack_scap_test.pkl"
    # --- Solve the program --- #
    myocp = UpperLimbOCP(
        biorbd_model_path=model_path,
        n_shooting=50,
        ode_solver=ode_solver,
        n_threads=n_threads,
        dynamics_fcn=DynamicsFcn.MUSCLE_DRIVEN,
        slack=1,
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
