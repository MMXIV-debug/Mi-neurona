from neuron import Neuron

class Network:

    def __init__(self):
        self.hidden = [Neuron(2), Neuron(2)]
        self.output = Neuron(2)
    
    def forward(self,inputs):
        hidden_outputs = []
        for neuron in self.hidden:
            hidden_outputs.append(neuron.forward(inputs))
        output = self.output.forward(hidden_outputs)
        return output