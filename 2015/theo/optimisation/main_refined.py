__author__ = 'theopavlakou'

import numpy as np
from cost_functions.Squared_Loss import Squared_Loss
from algorithms.Gradient_Descent import Gradient_Descent as GD
from data_creation.Boolean_Data_Creator import Boolean_Data_Creator

"""
Why is it better:
    1.  Documentation: Every class has pretty clear documentation.
    2.  Competely generalisable: I can change the algorithm and the
        cost function is seconds, literally.
    3.  Organised directory structure. I can find everything I want.
    4.  Less than 40 lines of code (and could easily be further refactored).
"""

# Initial set up that was there before.
N = 100
d = 2
iterations = 100
lmda = 1e-5

bdc = Boolean_Data_Creator()
data = bdc.create_data(N, d)
w_init = np.random.randn(d, 1)

cost_f = Squared_Loss(data.X, data.Y, lmda)
gd = GD(cost_f, iterations, w_init=w_init, step_size=0.01)

# Could go even further and make the optimisation algorithm
# return an Optimisation_Result which would could be saved in
# a pickle file etc.
w_new, costs = gd.optimise_function()

print("The optimised w parameter is \n{0}".format(w_new))
print("The initial cost was {0}".format(costs[0]))
print("The final cost is {0}".format(costs[-1]))

