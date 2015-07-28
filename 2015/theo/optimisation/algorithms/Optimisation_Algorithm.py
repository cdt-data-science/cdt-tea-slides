__author__ = 'theopavlakou'

import numpy as np
rng = np.random

class Optimisation_Algorithm(object):
    """
    An abstract class that provides the interface for
    optimisation algorithms.

    It ensures that the algorithm knows what cost function it
    will be working on (which includes the data also). It also
    defines the maximum number of iterations for the algorithm
    to be run.
    """

    def __init__(self, cost_f, max_it, w_init=None, step_size=0.01):
        """
        Initialises the optimisation algorithm.

        :param cost_f:  The function that the algorithm will minimise.
                        This should be of type optimisation.Cost_Function.
        :param max_it:  The maximum number of iterations the algorithm
                        will be run for.
        :param w_init:  Initial parameter vector. Must have the same
                        dimension as cost_f.dimension.
        :param step_size:   The step size to be used for the update
                            by the algorithm.
        """
        self.cost_f = cost_f
        self.max_it = max_it
        self.d = self.cost_f.dimension
        self.num = self.cost_f.num
        self.l = self.cost_f.l
        if w_init is None:
            self.w = rng.randn(self.d, 1)
        else:
            self.w = w_init
        self.step_size = step_size

    def do_iteration(self):
        """
        This will be done for every iteration.
        It updates the parameter vector, self.w.
        This is specific to the concrete implementation.
        """
        pass

    def optimise_function(self):
        """
        The template for optimising a function.
        As it can be seen, the only thing that
        needs to change in this whole algorithm
        is the self.do_iteration() method.

        :return: self.w: the final parameter vector
        :return: cost_list: a list of costs after
                            each iteration.
        """
        counter = 0
        cost_list = [self.cost_f.loss(self.w)]
        while counter < self.max_it:
            self.do_iteration()
            cost_list.append(self.cost_f.loss(self.w))
            counter += 1
        return self.w, cost_list