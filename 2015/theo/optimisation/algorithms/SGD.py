__author__ = 'theopavlakou'

from Optimisation_Algorithm import Optimisation_Algorithm
import numpy as np
rng = np.random

class SGD(Optimisation_Algorithm):
    """
    A class that performs stochastic gradient descent to optimise a cost function.
    """

    def __init__(self, cost_f, max_it, w_init=None, step_size=0.01, minibatch_size=1):
        """
        Initialises SGD. This takes an extra parameter, the
        minibatch_size.

        :param cost_f:  The function that the algorithm will minimise.
                        This should be of type optimisation.Cost_Function.
        :param max_it:  The maximum number of iterations the algorithm
                        will be run for.
        :param w_init:  Initial parameter vector. Must have the same
                        dimension as cost_f.dimension.
        :param step_size:   The step size to be used for the update
                            by the algorithm.
        :param minibatch_size:  The size of the minibatch to be used
                                upon each iteration.
        """
        super(SGD, self).__init__(cost_f, max_it, w_init, step_size)
        self.step_size = step_size
        self.minibatch_size = minibatch_size

    def do_iteration(self):
        """
        Chooses a random subset of the data points and performs the
        stochastic gradient descent update.
        """

        indices = rng.randint(self.cost_f.num, size=self.minibatch_size)
        self.w -= self.step_size*self.cost_f.derivative(self.w, indices)

    def optimise_function(self):
        """
        Optimises the cost function with SGD.

        :return: self.w: the final parameter vector
        :return: cost_list: a list of costs after
                            each iteration.
        """
        return super(SGD, self).optimise_function()
