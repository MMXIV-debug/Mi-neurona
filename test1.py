
from brain.neuron import Neuron
from brain.vector import Vector
from brain.matrix import Matrix


# PRUEBA 1: ENTRENAMIENTO DE UNA NEURONA

print("\n========== PRUEBA DE NEURONA ==========")
neuron = Neuron(4)
dataset = [
    ([1, 1, 0, 0], 1),
    ([1, 0, 0, 0], 1),
    ([0, 1, 0, 0], 1),
    ([0, 0, 1, 0], 0),
    ([0, 0, 0, 1], 0),
    ([0, 0, 0, 0], 0)
]

for epoch in range(1000):
    total_error = 0
    for x, y in dataset:
        total_error += neuron.train(x, y)
    if epoch % 100 == 0:
        print("Epoca:", epoch)
        print("Error:", total_error)


print("\nPesos finales:")
print(neuron.weights)

print("Bias final:")
print(neuron.bias)


# PRUEBA 2: OPERACIONES CON VECTORES

print("\n========== PRUEBA DE VECTORES ==========")

# Suma
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
c = a.add(b)
print("Suma:", c.data)


# Resta
c = a.subtract(b)
print("Resta:", c.data)


# Multiplicacion por escalar
v = Vector([1, 2, 3])
result = v.multiply_scalar(2)
print("Multiplicacion escalar:", result.data)


# Magnitud
v = Vector([3, 4])
print("Magnitud:", v.magnitude())


# Normalizacion
v = Vector([3, 4])
normalized = v.normalize()
print("Vector normalizado:", normalized.data)


# Producto punto
a = Vector([1, 2, 3])
b = Vector([4, 5, 6])
print("Producto punto:", a.dot(b))


# PRUEBA 3: OPERACIONES CON MATRICES

print("\n========== PRUEBA DE MATRICES ==========")
m = Matrix([
    [1, 2, 3],
    [4, 5, 6]
])
print("Filas:", m.rows())
print("Columnas:", m.cols())

# Multiplicacion matriz-vector
v = Vector([7, 8, 9])
result = m.multiply_vector(v)
print("Multiplicacion matriz-vector:", result.data)


# Suma de matrices
m1 = Matrix([
    [1, 2],
    [3, 4]
])
m2 = Matrix([
    [5, 6],
    [7, 8]
])
result = m1.add(m2)
print("Suma de matrices:", result.data)


# Resta de matrices
result = m1.substract(m2)
print("Resta de matrices:", result.data)


# Multiplicacion por escalar
result = m1.multiply_scalar(2)
print("Multiplicacion escalar:", result.data)

print("\n========== TODAS LAS PRUEBAS FINALIZADAS ==========")
