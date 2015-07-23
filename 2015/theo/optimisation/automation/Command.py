__author__ = 'theopavlakou'

class Command(object):
    """
    The interface for the Command Pattern.
    """
    def execute(self):
        """
        Needs to be implemented by any class implementing
        this interface. Executes the command.
        """
        pass