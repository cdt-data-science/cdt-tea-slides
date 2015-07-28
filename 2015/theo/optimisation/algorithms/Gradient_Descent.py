__author__ = 'theopavlakou'

from Optimisation_Algorithm import Optimisation_Algorithm
import numpy as np
rng = np.random

class Gradient_Descent(Optimisation_Algorithm):
    """
    A class that performs gradient descent to optimise a cost function.
    """
    def __init__(self, cost_f, max_it, w_init=None, step_size=0.01):
        """
        Initialises Gradient_Descent.

        :param cost_f:  The function that the algorithm will minimise.
                        This should be of type optimisation.Cost_Function.
        :param max_it:  The maximum number of iterations the algorithm
                        will be run for.
        :param w_init:  Initial parameter vector. Must have the same
                        dimension as cost_f.dimension.
        :param step_size:   The step size to be used for the update
                            by the algorithm.
        """
        super(Gradient_Descent, self).__init__(cost_f, max_it, w_init, step_size)

    def do_iteration(self):
        """
        This will be done for every iteration.
        It updates the parameter vector, self.w.
        This is specific to Gradient_Descent.
        """
        self.w -= self.step_size*self.cost_f.derivative(self.w)

    def optimise_function(self):
        """
        Optimises the cost function with gradient descent.

        :return: self.w: the final parameter vector
        :return: cost_list: a list of costs after
                            each iteration.
        """
        return super(Gradient_Descent, self).optimise_function()
