
class RegularizerAbstraction:
    def __init__(self, regularizer):
        self._regularizer_factor = regularizer

    def regularizer(self, *args, **kwargs):
        return self._regularizer_factor * self._regularizer(*args, **kwargs)

    def _regularizer(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return "using %s with regularizer=%.1e" % (type(self).__name__, self._regularizer_factor)
