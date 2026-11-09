
from brain.layer import Layer
from brain.vector import Vector

# Crear una capa con 3 entradas y 4 neuronas
layer = Layer(3, 4)

# Crear vector de entrada
inputs = Vector([1, 0, 1])

# Ejecutar forward propagation
outputs = layer.forward(inputs)

print("PRUEBA DE CAPA NEURONAL")

print("\nEntradas:")
print(inputs.data)

print("\nMatriz de pesos:")
print(layer.weights.data)

print("\nBias:")
print(layer.biases.data)

print("\nSalidas:")
print(outputs.data)

print("\nCantidad de salidas:")
print(outputs.size())
