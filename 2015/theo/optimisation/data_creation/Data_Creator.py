__author__ = 'theopavlakou'

class Data_Creator(object):
    """
    An abstract class that sets the template of what
    a Data_Creator should do.
    """

    def create_data(self, N, d):
        """
        Creates the data. This needs to be overrided by
        any concrete class.

        :param N: The number of data points.
        :param d: The dimension of the input.
        :return:  a Data object.
        """
        pass