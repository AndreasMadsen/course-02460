
class StoppingAbstraction:
    """abstraction defining expected members for early_stopping
    """
    
    def __init__(self, *args, verbose=False, **kwargs):
         raise NotImplementedError()

    def is_converged(self, loss):
         raise NotImplementedError()
