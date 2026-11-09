import math
import random

from .matrix import Matrix
from .vector import Vector

class Layer:

    def __init__(self, input_size, neuron_count):
        self.input_size = input_size
        self.neuron_count = neuron_count

        # Inicializar matriz de pesos
        self.weights = Matrix([
            [
                random.uniform(-1, 1)
                for _ in range(input_size)
            ]
            for _ in range(neuron_count)
        ])

        # Inicializar bias de cada neurona
        self.biases = Vector([
            random.uniform(-1, 1)
            for _ in range(neuron_count)
        ])

        # Variables para almacenar resultados de Forward Propagation
        self.last_input = None
        self.last_z = None
        self.last_output = None

# Funcion de activacion Sigmoid
    def sigmoid(self, x):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1 + exp_x)
        
# Propagacion hacia adelante
    def forward(self, inputs):
        if inputs.size() != self.input_size:
            raise ValueError("El tamaño de la entrad no coincide con la capa")

        # Guardamos las entradas
        self.last_input = Vector(inputs.data[:])

        # Multiplicar pesos por entradas
        z = self.weights.multiply_vector(inputs)

        # sumar bias
        for i in range(self.neuron_count):
            z.data[i] += self.biases.data[i]

        # Guardar valores antes del sigmoid
        self.last_z = Vector(z.data[:])

    # aplicar funcion de activacion
        outputs = []
        for value in z.data:
            outputs.append(
                self.sigmoid(value)
            )

        # Guardar las salidas como un Vector
        self.last_output = Vector(outputs)
        
        # Retornar el resultado
        return self.last_output
