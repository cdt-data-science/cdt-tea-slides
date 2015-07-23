__author__ = 'theopavlakou'

from Boolean_Data_Creator import Boolean_Data_Creator
from Multi_Class_Data_Creator import Multi_Class_Data_Creator

class Data_Creator_Factory(object):
    """
    This conforms to the Factory Design Pattern. It creates
    Data_Creator concrete classes.
    """
    def create_data_creator(self, type):
        """
        Creates a Data_Creator concrete class.

        :param type:    a string which determines the
                        type of Data_Creator to create.
        :return: the Data_Creator requested.
        """
        if type == "boolean":
            return Boolean_Data_Creator()
        if type == "multi":
            return Multi_Class_Data_Creator()
        """
        Any time a new type of data creator is created it
        only needs to be added here. No need to have this code
        everywhere in your system. Now just add a string and
        a new class and that can be called from anywhere in your
        code i.e. when something changes, the only thing that
        needs to change is this code here, not in 120898 other
        places. 
        """