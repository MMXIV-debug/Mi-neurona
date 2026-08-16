import math
import random

class Vector:

    def __init__(self, data):
        self.data = data

    def size(self):
        return len(self.data)
    
    def add(self,other):
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i] + other.data[i])
        return Vector(result)
    
    def subtract(self,other):
        result = []
        for i in range(len(self.data)):
            result.append(self.data[i] - other.data[i])
        return Vector(result)
    
    def multiply_scalar(self,value):
        result = []
        for x in self.data:
            result.append(x * value)
        return Vector(result)
    
    def magnitude(self):
        total = 0
        for x in self.data:
            total += x*x
        return math.sqrt(total)
    
    def normalize(self):
        mag = self.magnitude()
        result = []
        for x in self.data:
            result.append(x / mag)
        return Vector(result)
    
    def dot(self,other):
        result = 0
        for i in range(len(self.data)):
            result += (self.data[i] * other.data[i])
        return result