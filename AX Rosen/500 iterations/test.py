
#questo codice fa una prova di uso dei thread, non serve per la BO

import rosen
from ReturnThread import ReturnThread

x = [1, 1]
a = 1
b = 100


function_thread = ReturnThread(target=rosen.function, args=([x[0], x[1], a, b]))
function_thread.start()
function_thread.join()



evaluated_function = function_thread.value
SEM = 0 #errore standard di misura, essendo una funzione analitica (computer) è 0
print( {"rosenbrock": (evaluated_function, SEM)})
