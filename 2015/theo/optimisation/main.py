__author__ = 'theopavlakou'

import numpy as np


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

N = 100
d = 2
iterations = 100000
lmda = 1e-5
X_neg = np.random.randn(np.floor(N/2.0), d) + 2*np.ones((1, d))
X_pos = np.random.randn(np.ceil(N/2.0), d) - 2*np.ones((1, d))
X_data = np.vstack((X_neg, X_pos))
Y_neg = np.zeros((np.floor(N/2.0), 1))
Y_pos = np.ones((np.ceil(N/2.0), 1))
Y = np.vstack((Y_neg, Y_pos))
w_init = np.random.randn(d, 1)
w_new, costs = optimise_logistic_loss(w_init, X_data, Y, lmda, iterations)
print(w_new)
print(costs[0])
print(costs[-1])

