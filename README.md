# shoulder-rhythm-ocp

This repository is a test for the ligaments in the bioMod and the functions to pilot the humeral elevation.

The model used is the "wu_converted_definitif_without_floating_base", with an added phantom segment so that all bones have one segment following the ISB recommendations.

The files in results are the ones to plot or animate the results of the main. 
The file "graph_results_q_upper_limb" plots the q, tau and muscles activations of chosen script(s)
The file "graph_results_angles_upper_limb" plots the ISB SC, ST, Glenohumeral and Acromioclavicular angles in function of the humeral elevation in the thoracic CS.
The file "Animate_from_data_pckl" is the script to animate the results of chosen pickle file.

The ocp repertory contains all the files used to run the simulations. 
The "main" file is only used to start the simulation. You can choose the humerus elevation, the slack, the dynamics, and the plots
The "upper_limb" file is the principal file of this repository, where all the ocp parameters are set (bounds, objectives and constraints functions, initial guess, ...)
It is in this one that you have to set the problem.
The "utils" file is the file where you will find all the functions that are used to help set the problem, like custom objectives and dynamics, interface functions, result saving function, ...

The "guided_elevation_results" is the repertory where the pickle of a good-looking animation from a guided elevation of the humerus is stored