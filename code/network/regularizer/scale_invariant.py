
import lasagne
import theano.tensor as T
import theano

from network.regularizer.abstaction import RegularizerAbstraction

class ScaleInvariant(RegularizerAbstraction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _regularizer(self, prediction, input_var, target_var, **kwargs):
        # - Why not T.grad(loss[i], x[i, :, :, :]) instread of
        #   T.grad(loss[i], x)[i, :, :, :]). See:
        #   https://groups.google.com/forum/#!topic/theano-users/KmgNkAZsZPk
        # - T.grad of loss will produce CrossentropyCategorical1HotGrad
        #   however this does not have a defined gradient.
        #   https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/nnet.py#L1379
        #   Thus the log properbility is calculted explicitly, using
        #   T.log(p[i, t[i]])
        def loop(i, x, p, t):
            p_class_t = p[i, t[i]]

            return T.dot(
                T.flatten(T.grad(p_class_t, x)[i, :, :, :]),
                T.flatten(x[i, :, :, :])
            )

        jacobi_dot_x, _ = theano.scan(
            loop,
            non_sequences=[input_var, prediction, target_var],
            sequences=T.arange(input_var.shape[0])
        )

        return T.pow(jacobi_dot_x, 2).mean()
