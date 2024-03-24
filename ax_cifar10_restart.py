#!/usr/bin/env python
# coding: utf-8

from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.modelbridge.registry import ModelRegistryBase, Models
from ax.service.ax_client import AxClient, ObjectiveProperties
import pandas as pd
from ReturnThread import ReturnThread
import cifar10
import sys

def print_trials(trials): #print the trials
    for trial in trials:
        print(trial)

#load from stats.csv all the trials and their function value
#the data is in form ,trial_index,arm_name,trial_status,generation_method,total_accuracy,layer1,layer2,layer3,layer4,learning_rate
def load_from_stats(filename):
    df = pd.read_csv(filename)
    trials = list()
    for index, row in df.iterrows():
        if row["trial_status"] == "COMPLETED":
            parameters = {"layer1": row["layer1"], "layer2": row["layer2"], "layer3": row["layer3"], "layer4": row["layer4"], "learning_rate": str(row["learning_rate"])}
            function = (row["total_accuracy"], 0)
            trials.append((row["trial_index"], parameters, function))
    return trials

def get_evaluation(trials, parameters):
    for trial in trials:
        if trial[1] == parameters:
            return trial[2]
    return None

def evaluate(parameters): #computazione della funzione

    layer1 = parameters.get("layer1") * 12 #fattore moltiplicativo neuroni
    layer2 = parameters.get("layer2") * 12
    layer3 = parameters.get("layer3") * 12
    layer4 = parameters.get("layer4") * 12
    learning_rate = parameters.get("learning_rate") #learning rate

    hyperparameters = [layer1, layer2, layer3, layer4, learning_rate] #lista iperparametri

    function_thread = ReturnThread(target=cifar10.main, args=(hyperparameters)) #thread per la funzione di allenamento (black box expensive to evaluate)
    function_thread.start()
    function_thread.join() #attesa del thread

    accuracies = function_thread.getValue() #valore di ritorno del thread, un dizionario avente le varie accuratezze

    totaL_accuracy = accuracies['total_accuracy'] #ci interessa solo l'accuratezza totale
    SEM = 0 #errore standard di misura, essendo una funzione analitica (computer) è 0
    return {"total_accuracy": (totaL_accuracy, SEM)}

if __name__ == '__main__':
    VERBOSE = True #verbose mode
    RESTART_MODE = False #restart mode, se True carica i trials da stats.csv e prosegue, altrimenti inizia da zero

    gen_steps = [] #steps per la generazione
    
    if RESTART_MODE == False: #se non siamo in restart mode aggiungiamo sobol
        gen_steps.append(
            GenerationStep(
            model=Models.SOBOL,
            num_trials= 10,  # quanti trial produce la sequenza di sobol 
            min_trials_observed=8,  # quanti devono avere successo (ci lasciamo un po' di margine)
            max_parallelism=None,  # niente parallelismo
            model_kwargs={"seed": 999},  # Any kwargs you want passed into the model
            model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
            )
        )
    
    gen_steps.append( 
            GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,  # No limitation on how many trials should be produced from this step
            max_parallelism=None,  # Parallelism limit for this step, often lower than for Sobol
            # More on parallelism vs. required samples in BayesOpt:
            # https://ax.dev/docs/bayesopt.html#tradeoff-between-parallelism-and-total-number-of-trials
        )
    )

    print(gen_steps)

    gs = GenerationStrategy(steps=gen_steps) #creazione della strategia di generazione
    
    ax_client = AxClient(enforce_sequential_optimization=True, generation_strategy=gs) #creazione del client, forziamo l'ottimizzazione sequenziale

    ax_client.create_experiment( 
        name="cifar10",
        parameters=[
            {
                "name": "layer1",
                "type": "range",
                "bounds": [1, 16],
                "value_type": "int",  # Opzionale se non specificato viene inferito dal tipo di "bounds".
                "log_scale": False,  # Opzionale, default False.

            },
            {
                "name": "layer2",
                "type": "range",
                "bounds": [1, 16],
                "value_type": "int",  
                "log_scale": False,  
            },
            {
                "name": "layer3",
                "type": "range",
                "bounds": [1, 16],
                "value_type": "int",  
                "log_scale": False, 
            },
            {
                "name": "layer4",
                "type": "range",
                "bounds": [1, 16],
                "value_type": "int",  
                "log_scale": False,  
            },
            {
                "name": "learning_rate",
                "type": "choice", #il tipo di variabile è una scelta tra i values
                "value_type": "str",  #sono stringhe
                "values": ["0.01", "0.001", "0.0001"], #valori accettabili in scala logaritmica del learning_rate
                "is_ordered": True,
            },
        ],
        objectives={"total_accuracy": ObjectiveProperties(minimize=False)}, #funzione da massimizzare
        #parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
        #outcome_constraints=["l2norm <= 1.25"],  # Optional.
    )


    ax_client.save_to_json_file("./cifar10.json") #salvataggio del client in un file json
    ax_client.load_from_json_file("./cifar10.json")

    evaluated_trials = list()

    if RESTART_MODE == True: #se siamo in restart mode carichiamo i trials da stats.csv
        evaluated_trials = load_from_stats("./stats.csv")
    num_trials = 50

    best_trial = None

    trials = list()

    for i in range(num_trials + len(evaluated_trials)):

        if evaluated_trials != []:
            trial = evaluated_trials.pop(0)
            print(trial)
            parameters, trial_index = ax_client.attach_trial(parameters=trial[1])
            evaluation ={'total_accuracy': trial[2]}
    
        elif RESTART_MODE == False and i == 0:
            parameters, trial_index = ax_client.attach_trial(parameters={"layer1": 1, "layer2": 1, "layer3": 2, "layer4": 2, "learning_rate": "0.001"})
            evaluation = evaluate(parameters)
        else:
            parameters, trial_index = ax_client.get_next_trial()
            
            evaluation = get_evaluation(trials, parameters)
            if evaluation == None:
                evaluation = evaluate(parameters)

        if best_trial == None or evaluation["total_accuracy"][0] < best_trial[2]["total_accuracy"][0]: #aggiorna il best trial
            best_trial = (i, parameters, evaluation)

        trials.append( (i, parameters, evaluation) )

        
        ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation)

    ax_client.get_trials_data_frame().to_csv("./stats.csv")

    print("best_trial: ")
    print(best_trial)

    
    if VERBOSE == True: #verbose mode stampa i risultati
        print("VERBOSE MODE: ")
        best_parameters, values = ax_client.get_best_parameters()
        print("best_parameters:" + str(best_parameters))
        print("function at best parameters:" + str(evaluate(best_parameters)))

        print("trials: ")
        print_trials(trials)

        means, covariances = values
        print("means:" + str(means))
        print("covariences: " + str(covariances))

    sys.exit(0)
