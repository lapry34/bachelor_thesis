#!/usr/bin/env python
# coding: utf-8


from ax.service.ax_client import AxClient, ObjectiveProperties
from ReturnThread import ReturnThread
import sys
import numpy as np
import fungoldprice as goldprice

def print_trials(trials):
    for trial in trials:
        print(trial)


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(len(parameters))])

    function_thread = ReturnThread(target=goldprice.goldstein_price_function, args=([x[0], x[1]]))
    function_thread.start()
    function_thread.join()

    evaluated_function = function_thread.getValue()
    SEM = 0 #errore standard di misura, essendo una funzione analitica (computer) è 0
    return {"function": (evaluated_function, SEM)}

if __name__ == '__main__':
    ax_client = AxClient(enforce_sequential_optimization=True)
    VERBOSE = True

    ax_client.create_experiment( 
        name="Goldprice",
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [-2, 10],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".

            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [-10, 2],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            },
           
        ],
        objectives={"function": ObjectiveProperties(minimize=True)},

    )


    ax_client.save_to_json_file("./Goldprice.json")
    ax_client.load_from_json_file("./Goldprice.json")

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

    ax_client.get_trials_data_frame().to_csv("./stats.csv")

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
