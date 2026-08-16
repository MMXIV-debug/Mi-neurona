import math
import random

class Neuron:

    def __init__(self, input_size):
        #pesos y bias
        self.weights = []
        for _ in range(input_size):
            self.weights.append(random.uniform(-1,1))
        self.bias = random.uniform(-1,1)

    def sigmoid(self,x):
        #funcion de activacion
        return 1 / (1 + math.exp(-x))
    
    def forward(self, inputs):
        #calculo de la salida de la neurona
        z = self.bias
        for i in range(len(inputs)):
            z += (inputs[i] * self.weights[i])
        return self.sigmoid(z)
    
    def train(self, inputs, expected, learning_rate=0.1):
        #ajuste de pesos y bias usando el error y la derivada de la funcion de activacion
        predicted = self.forward(inputs)
        error = expected - predicted
        gradient = error * predicted * (1 - predicted)
        for i in range(len(self.weights)):
            self.weights[i] += (learning_rate * gradient * inputs[i])
        self.bias += (learning_rate * gradient)
        return abs(error)