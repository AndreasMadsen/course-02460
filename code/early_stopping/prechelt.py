import numpy as np
from early_stopping.abstraction import StoppingAbstraction

class PrecheltStopping(StoppingAbstraction):
    """ Implements a variation on  the second method described in 
    http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    (see pages 4-5)
    """
    
    def __init__(self, *args, alpha=5/2, interval_length=25,
                 verbose=False, **kwargs):
        self.losses = list()
        self.alpha = alpha
        self.interval_length = interval_length
        self.verbose = verbose
        if verbose:
            print("Initializing early stopping class Prechelt. ")
            print("Criterion is [generalization loss] / [improvement factor] > alpha ")
            print("\talpha = {0}".format(alpha))
            print("\timprovement factor length = {0}".format(interval_length))


    def is_converged(self, loss):
        
        # find the lowest loss encountered so far       
        if len(self.losses) == 0:
            lowest_loss_val = np.inf
        else:
            lowest_loss_val = np.min(self.losses)
        
        # update liss of losses
        self.losses.append(loss)
        epoch = len(self.losses)
        
        # compute factors
        generalization_loss = self.losses[-1] / lowest_loss_val - 1
        if epoch < self.interval_length:
            improvement_factor = 1
        else:
            interval = self.losses[(epoch - self.interval_length):epoch]
            improvement_factor = np.mean(interval) / np.min(interval) - 1

        criteria = generalization_loss / improvement_factor
        
        if self.verbose:
            print("Generalization_loss = {0:.3f}".format(generalization_loss),end="")
            print(". Improvement factor = {0:.3f}".format(improvement_factor), end="")
            print(". GL / IF = {0:.3f}".format(criteria))

        if criteria > self.alpha:
            print("stopping criteria: {0:.3f} > {1:.3f} is {2}".format(criteria,
                                                                       self.alpha,
                                                                       True))
            return True
        else:
            return False
