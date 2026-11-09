from .vector import Vector


class Matrix:

    def __init__(self, data):

        if len(data) == 0:
            raise ValueError("La matriz no puede estar vacia")

        columns = len(data[0])

        if columns == 0:
            raise ValueError("La matriz debe tener columnas")

        for row in data:

            if len(row) != columns:
                raise ValueError("Todas las filas deben tener el mismo tamaño")

        self.data = [row[:] for row in data]

    def rows(self):
        return len(self.data)

    def cols(self):
        return len(self.data[0])

    def add(self, other):
        if self.rows() != other.rows() or self.cols() != other.cols():
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        result = []
        for r in range(self.rows()):
            row = []
            for c in range(self.cols()):
                row.append(
                    self.data[r][c] +
                    other.data[r][c]
                )
            result.append(row)
        return Matrix(result)

    def subtract(self, other):
        if self.rows() != other.rows() or self.cols() != other.cols():
            raise ValueError("Las matrices deben tener las mismas dimensiones")
        result = []
        for r in range(self.rows()):
            row = []
            for c in range(self.cols()):
                row.append(
                    self.data[r][c] -
                    other.data[r][c]
                )
            result.append(row)
        return Matrix(result)

    def multiply_scalar(self, value):
        result = []
        for row in self.data:
            new_row = []
            for x in row:
                new_row.append(x * value)
            result.append(new_row)
        return Matrix(result)
    
    def multiply_vector(self, vector):
        if self.cols() != vector.size():
            raise ValueError("Dimensiones incompatibles para multiplicar")
        result = []
        for row in self.data:
            total = 0
            for i in range(self.cols()):
                total += (
                    row[i] *
                    vector.data[i]
                )
            result.append(total)
        return Vector(result)