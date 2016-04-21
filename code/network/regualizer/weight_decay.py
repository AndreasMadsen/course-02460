
import lasagne
import theano.tensor as T

from network.regualizer.abstaction import RegualizerAbstraction

class WeightDecay(RegualizerAbstraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _regualizer(self, network, **kwargs):
        L2 = lasagne.regularization.regularize_network_params(
            network,
            lasagne.regularization.l2
        )
        return L2
