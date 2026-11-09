
from .layer import Layer

class Network:

    def __init__(self, architecture):
        # Comprobar que exista una entrada y una capa de salida
        if len(architecture) < 2:
            raise ValueError(
                "La red necesita al menos dos dimensiones"
            )
        # Comprobar dimensiones
        for size in architecture:
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(
                    "Las dimensiones deben ser enteros positivos"
                )
        self.architecture = architecture[:]
        # Lista de capas
        self.layers = []
        # Construir las capas automaticamente
        for i in range(len(architecture) - 1):
            input_size = architecture[i]
            neuron_count = architecture[i + 1]
            layer = Layer(input_size, neuron_count)
            self.layers.append(layer)

    # Forward Propagation de toda la red
    def forward(self, inputs):
        if inputs.size() != self.architecture[0]:
            raise ValueError(
                "El tamaño de entrada no coincide con la arquitectura"
            )
        # Las entradas iniciales
        current = inputs
        # Recorrer todas las capas
        for layer in self.layers:
            current = layer.forward(current)
        # Devolver resultado final
        return current
