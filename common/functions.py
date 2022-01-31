import numpy as np
from scipy.optimize import root


def Γ(y):
    """Vandenkerckhove function"""
    return y**0.5 * (2 / (y + 1)) ** ((y + 1) / (2 * y - 2))


def exit_pressure(y, AeAt, Pc):
    """Calculates the exit pressure using just floats"""
    term1 = ((2 * y) / (y - 1))
    vdkerckhove = Γ(y)

    def find_value():
        """Finds the value of Pe"""
        def f(Pe):
            PePc = Pe / Pc
            term2 = (PePc ** (2 / y))
            term3 = 1 - PePc ** ((y - 1) / y)
            return vdkerckhove / (np.sqrt(term1 * term2 * term3)) - AeAt

        # Relatively close guess that I found that works for y=1.01...1.5 and Ae/At=3..20
        guess = (Pc/2) / AeAt ** (y + 0.5)

        # Start root finding function
        ans = root(f, guess)
        return ans.x[0]

    value = find_value()
    return value
