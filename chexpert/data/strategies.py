import numpy as np


class UIgnoreStrategy:
    def __call__(self, y):
        return y


class UOnesStrategy:
    def __call__(self, y):
        y[y == -1] = 1
        return y


class UZerosStrategy:
    def __call__(self, y):
        y[y == -1] = 0
        return y


class UUniformStrategy:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, y):
        mask = np.random.uniform(self.low, self.high, size=(y == -1).sum())
        y[y == -1] = mask
        return y


class UBetaStrategy:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self, y):
        mask = np.random.beta(self.a, self.b, size=(y == -1).sum())
        y[y == -1] = mask
        return y
