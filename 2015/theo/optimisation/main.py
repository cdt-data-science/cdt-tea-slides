__author__ = 'theopavlakou'

import numpy as np

"""
Problems:
    1.  No documentation: It's hardly clear to me what some things mean.
    2.  Not generalisable: See optimise_logistic_loss(). Suppose I want
        to optimise for a squared loss cost function, more than half the
        code would be exactly the same. Or suppose I want to use another
        algorithm for the algorithms.
    3.  It is completely unorganised.
    4.  Almost 100 lines of code for an algorithm that is so simple.
"""


def sigma(w, X):
    return 1.0 / (1 + np.exp(-np.dot(X, w)))


def logistic_loss(w, X, y, l):

    soft_pred = sigma(w, X)
    zero = (soft_pred == 0)
    soft_pred[zero] = np.finfo(float).eps
    a = np.log(soft_pred)

    # This is purely to ensure we don't take log of 0.
    # Bad things happen when we take the log of 0.
    b = np.log(1 + np.finfo(float).eps - soft_pred)

    return np.mean(-y * a - (1 - y) * b) + l / 2 * (np.sum(w ** 2))


def derivative_logistic_loss(w, X, y, l):
    return l*w + np.dot(X.T, (sigma(w, X) - y))/X.shape[0]



def square_loss(w, X, y, l):
    f_w = 0.5*np.mean((np.dot(X, w) - y)**2)
    return f_w + l / 2.0 * (np.sum(w ** 2))


def derivative_square_loss(w, X, y, l):
    return l*w + 1.0/X.shape[0]*np.dot(X.T, (np.dot(X, w) - y))


def loss_3():
    pass


def derivative_loss_3():
    pass


def optimise_logistic_loss(w_init, X, y, l, max_it):
    i = 0
    eta = 0.03
    w = w_init
    cost_list = [logistic_loss(w, X, y, l)]
    while i < max_it:
        w -= eta*derivative_logistic_loss(w, X, y, l)
        cost_list.append(logistic_loss(w, X, y, l))
        i += 1
    return w, cost_list


def optimise_squared_loss(w_init, X, y, l, max_it):
    i = 0
    eta = 0.01
    w = w_init
    cost_list = [logistic_loss(w, X, y, l)]
    while i < max_it:
        w -= eta*derivative_square_loss(w, X, y, l)
        cost_list.append(square_loss(w, X, y, l))
        i += 1
    return w, cost_list


def optimise_loss_3():
    pass

N = 100
d = 2
iterations = 100000
lmda = 1e-5

w_new, costs = optimise_squared_loss(w_init, X_data, Y, lmda, iterations)

print("The optimised w parameter is \n{0}".format(w_new))
print("The initial cost was {0}".format(costs[0]))
print("The final cost is {0}".format(costs[-1]))

