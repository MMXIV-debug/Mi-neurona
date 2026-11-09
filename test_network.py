from brain.network import Network
from brain.vector import Vector

# Crear nuestra red neuronal
network = Network([5, 8, 4])

# Crear vector de entrada
inputs = Vector([1, 0, 1, 0, 1])

# Ejecutar Forward Propagation
outputs = network.forward(inputs)

print("PRUEBA DE RED NEURONAL")

print("\nArquitectura:")
print(network.architecture)

print("\nCantidad de capas:")
print(len(network.layers))

# Mostrar informacion de cada capa
for i in range(len(network.layers)):
    layer = network.layers[i]
    print("\nLAYER", i + 1, "")

    print("Entradas:", layer.input_size)

    print("Neuronas:", layer.neuron_count)

    print("Pesos:")
    print(layer.weights.data)

    print("Bias:")
    print(layer.biases.data)

    print("Entrada recibida:")
    print(layer.last_input.data)

    print("Resultado antes de Sigmoid:")
    print(layer.last_z.data)

    print("Resultado despues de Sigmoid:")
    print(layer.last_output.data)


print("\n========== RESULTADO FINAL ==========")

print(outputs.data)

print("\nCantidad de salidas:")
print(outputs.size())