import numpy as np


class Meter(object):
    def reset(self):
        pass

    def add(self, value):
        pass

    def value(self):
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.n = 0
        self.mean = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.n += n
        self.mean = self.mean + (value - self.mean) / self.n

