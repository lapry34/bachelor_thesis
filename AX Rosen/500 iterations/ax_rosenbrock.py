#!/usr/bin/env python
# coding: utf-8


from ax.service.ax_client import AxClient, ObjectiveProperties
from ReturnThread import ReturnThread
import rosen
import sys
import numpy as np

def print_trials(trials):
    for trial in trials:
        print(trial)


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(len(parameters))])
    a = 1
    b = 100

    function_thread = ReturnThread(target=rosen.function, args=([x[0], x[1], a, b]))
    function_thread.start()
    function_thread.join()

    evaluated_function = function_thread.getValue()
    SEM = 0 #errore standard di misura, essendo una funzione analitica (computer) è 0
    return {"function": (evaluated_function, SEM)}

if __name__ == '__main__':
    ax_client = AxClient(enforce_sequential_optimization=True)
    VERBOSE = False

    ax_client.create_experiment( 
        name="rosenbrock",
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [-100.0, 100.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".

            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [-100.0, 100.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            },
        ],
        objectives={"function": ObjectiveProperties(minimize=True)},
        #parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
        #outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )


    ax_client.save_to_json_file("./rosenbrock.json")
    ax_client.load_from_json_file("./rosenbrock.json")

    num_trials = 500

    best_trial = None

    trials = list()

    for i in range(num_trials):
        parameters, trial_index = ax_client.get_next_trial()

        evaluation = evaluate(parameters)

        if best_trial == None or evaluation["function"][0] < best_trial[2]["function"][0]: #aggiorna il best trial
            best_trial = (i, parameters, evaluation)

        trials.append( (i, parameters, evaluation) )
        
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation)

    ax_client.get_trials_data_frame().to_csv("./rosenbrock.csv")

    print(best_trial)

    if VERBOSE == True:
        best_parameters, values = ax_client.get_best_parameters()
        print("best_parameters:" + str(best_parameters))
        print("function at best parameters:" + str(evaluate(best_parameters)))

        print("trials: ")
        print_trials(trials)

        means, covariances = values
        print("means:" + str(means))
        print("covariences: " + str(covariances))

    sys.exit(0)
