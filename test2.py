from neuronS import SimpleNeuron
n = SimpleNeuron()
#Ejecuta el procedimiento de entrenamiento con epoch unas 100 veces, con un valor de entrada de 1 y un valor esperado de 1
#Luego imprime el peso actualizado después del entrenamiento
for epoch in range (100):
    error = n.train(
        x=1,
        expected=1
    )
print(n.weight)