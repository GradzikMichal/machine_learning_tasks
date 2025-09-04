from typing import Any
import numpy as np


class Activation:

    def apply(self, x):
        """
        Applies the activation function
        """
        pass

    def derivative(self, x):
        pass

    def __getitem__(self, x):
        return self.derivative(x)

    def __call__(self, x) -> Any:
        return self.apply(x)


############################

class ReLu(Activation):

    def apply(self, x):
        return np.where(x > 0, x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __getitem__(self, x):
        return self.derivative(x)

    def __call__(self, x) -> Any:
        return ReLu.apply(x)


############################

class LogSoftMax(Activation):

    def apply(self, x):
        #x = x - np.amax(x, axis=0)
        #return np.exp(x) / np.sum(np.exp(x), axis=0)
        out = np.exp(x)
        # freaking numpy!...
        suma = np.sum(out, axis=1)
        # print("exp", out.shape)
        # print("Sum", suma.shape)
        return np.divide(out.T, suma).T

    def derivative(self, x):
        SumE = np.sum(np.e ** x)
        sqSumE = np.power(SumE, 2)
        y = []
        for x_i in x:
            e_x = np.e ** x_i
            mul = e_x * (SumE - e_x)
            y.append(mul / sqSumE)
        return y


########################
class Sigmoid(Activation):
    def apply(self, x):
        return 1 / (1 + np.e ** (-x))

    def derivative(self, x):
        return (1 - self.apply(x)) * self.apply(x)

class Step(Activation):
    def apply(self, x):
        return np.where(x > 0, 1, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def __getitem__(self, x):
        return self.derivative(x)

    def __call__(self, x) -> Any:
        return ReLu.apply(x)

    def derivative(self, x):
        return 1 - np.tanh(x)**2