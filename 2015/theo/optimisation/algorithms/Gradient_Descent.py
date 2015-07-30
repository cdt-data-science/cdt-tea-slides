__author__ = 'theopavlakou'

from Optimisation_Algorithm import Optimisation_Algorithm
import numpy as np
rng = np.random

class Gradient_Descent(Optimisation_Algorithm):
    """
    A class that performs gradient descent to optimise a cost function.
    """
    def do_iteration(self):
        """
        This will be done for every iteration.
        It updates the parameter vector, self.w.
        This is specific to Gradient_Descent.
        """
        self.w -= self.step_size*self.cost_f.derivative(self.w)


