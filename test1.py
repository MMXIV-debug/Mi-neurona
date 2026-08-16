from brain.neuron import Neuron
from brain.vector import Vector

n = Neuron(4)
dataset = [([1,1,0,0],1),([1,0,0,0],1),([0,1,0,0],1),([0,0,1,0],0),([0,0,0,1],0),([0,0,0,0],0)]
for epoch in range(1000):
    total_error = 0
    for x,y in dataset:
        total_error += (
            n.train(x,y)
        )
    if epoch % 100 == 0:
        print(
            epoch,
            total_error
        )

a = Vector([1,2,3])
b = Vector([4,5,6])

c = a.add(b)

print(c.data)

v = Vector([1,2,3])

r = v.multiply_scalar(2)

print(r.data)

v = Vector([3,4])

print(v.magnitude())

v = Vector([3,4])

n = v.normalize()

print(n.data)

a = Vector([1,2,3])

b = Vector([4,5,6])

print(
    a.dot(b)
)