__author__ = 'theopavlakou'

from algorithms.Gradient_Descent import Gradient_Descent

class Gradient_Descent_Optimiation_Command(object):
    """
    Implements the Command interface. Encapsulates the
    Gradient Descent algorithm with its cost function and
    hyperparameters.
    """
    def __init__(self, cost_function, iterations, w_init=None, step_size=0.01):
        """
        Initialises the object.

        :param cost_function: the cost function to be optimised.
        :param iterations: the maximum number of iterations to run for.
        :param w_init: the initial parameter vector.
        :param step_size: the learning rate of the algorithm.
        """
        self.gd = Gradient_Descent(cost_function, iterations, w_init, step_size)

    def execute(self):
        """
        Executes optimisat_function() of gradient descent
        with the cost function and the hyperparameters provided.
        """
        return self.gd.optimise_function()