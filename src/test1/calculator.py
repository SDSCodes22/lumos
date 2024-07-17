import math

class Calculator:
    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def subtract(a, b):
        return a - b

    @staticmethod
    def multiply(a, b):
        return a * b

    @staticmethod
    def divide(a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    @staticmethod
    def square(a):
        return a ** 2

    @staticmethod
    def cube(a):
        return a ** 3

    @staticmethod
    def square_root(a):
        if a < 0:
            raise ValueError("Cannot take the square root of a negative number.")
        return math.sqrt(a)

    @staticmethod
    def cube_root(a):
        return a ** (1 / 3)

def main():
    calc = Calculator()

    # Example usage:
    print("Addition (5 + 3):", calc.add(5, 3))
    print("Subtraction (5 - 3):", calc.subtract(5, 3))
    print("Multiplication (5 * 3):", calc.multiply(5, 3))
    print("Division (5 / 3):", calc.divide(5, 3))
    print("Square (5^2):", calc.square(5))
    print("Cube (5^3):", calc.cube(5))
    print("Square Root (sqrt(25)):", calc.square_root(25))
    print("Cube Root (cbrt(27)):", calc.cube_root(27))

if __name__ == "__main__":
    main()
