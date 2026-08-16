import random

class SimpleNeuron:
    def __init__(self):
        #pesos y bias
        self.weight = random.random()
    def predict(self, x):
        #calculo de la salida
        z = self.weight * x
        return z
    def train(self, x, expected):
        #calculo de entrenamiento
        predicted = self.predict(x)
        error = expected - predicted
        learning_rate = 0.1
        self.weight += learning_rate * error * x
        return error