
import lasagne
import theano.tensor as T

from network.regularizer.abstaction import RegularizerAbstraction

class WeightDecay(RegularizerAbstraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _regularizer(self, network, **kwargs):
        L2 = lasagne.regularization.regularize_network_params(
            network,
            lasagne.regularization.l2
        )
        return L2
