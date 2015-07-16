__author__ = 'theopavlakou'
import unittest
from cost_functions.Squared_Loss import Squared_Loss
import numpy as np

class Test_Squared_Loss(unittest.TestCase):

    def setUp(self):
        pass

    def test_loss_no_regularisation(self):
        """
        Test that given the right parameters and a zero
        regularisation constant, the cost is equal to 0.
        """
        X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y = np.array([3, 6, 9])
        l = 0.0
        sl = Squared_Loss(X, y, l)
        w = np.array([1, 1, 1])
        self.assertEqual(sl.loss(w), 0.0)

    def test_loss_with_regularisation(self):
        """
        Test that given the right parameters and a
        regularisation constant, the cost is correct.
        """
        X = np.array([[1, 1, 1], [2, 2, 2]])
        y = np.array([3, 6])
        l = 1.0
        sl = Squared_Loss(X, y, l)
        w = np.array([1, 1, 1])
        self.assertEqual(sl.loss(w), 1.5)

    def test_loss_with_subset_of_data_points(self):
        """
        Test that the correct cost is given when a
        subset of the data points are used.
        """
        X = np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]])
        y = np.array([3, 6, 9])
        l = 0.0
        sl = Squared_Loss(X, y, l)
        w = np.array([1, 1, 1])
        indices = np.array([0, 2])
        self.assertEqual(sl.loss(w, indices), 2.25)

    def test_gradient_with_all_data_points_no_reg(self):
        """
        Tests that the gradient is given correctly with
        no regularisation constant and all the data points.
        """
        X = np.array([[0.5, 0.5, 0.5], [2, 2, 2], [3, 3, 3]])
        y = np.array([3, 6, 9])
        l = 0.0
        sl = Squared_Loss(X, y, l)
        w = np.array([1, 1, 1])
        self.assertEqual(sl.derivative(w).tolist(), [-0.25, -0.25, -0.25])

    def test_gradient_with_all_data_points_with_reg(self):
        """
        Tests that the gradient is given correctly with
        a regularisation constant and all the data points.
        """
        X = np.array([[0.5, 0.5, 0.5], [2, 2, 2], [3, 3, 3]])
        y = np.array([3, 6, 9])
        l = 1.0
        sl = Squared_Loss(X, y, l)
        w = np.array([1, 1, 1])
        self.assertEqual(sl.derivative(w).tolist(), [0.75, 0.75, 0.75])

    def test_gradient_with_subset_of_data_points_no_reg(self):
        """
        Tests that the gradient is given correctly with
        no regularisation constant and a subset of the data points.
        """
        # TODO
        pass

    def test_gradient_with_subset_of_data_points_with_reg(self):
        """
        Tests that the gradient is given correctly with
        regularisation constant and a subset of the data points.
        """
        # TODO
        pass