
class RegualizerAbstraction:
    def __init__(self, regualizer):
        self._regualizer_factor = regualizer

    def regualizer(self, prediction, network):
        return self._regualizer_factor * self._regualizer(prediction, network)

    def _regualizer(self, prediction, network):
        raise NotImplementedError

    def __str__(self):
        return "using %s with regualizer=%.1e" % (type(self).__name__, self._regualizer_factor)
