
import lasagne
import theano.tensor as T
import theano

from network.regularizer.abstaction import RegularizerAbstraction

class RotationInvariant(RegularizerAbstraction):
    def __init__(self, *args, use_Rop=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_Rop = use_Rop

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
                T.flatten(T.grad(p_class_t, x)[i]),
                T.flatten(x[i])
            )

        if self._use_Rop:
            (x, p, t) = (input_var, prediction, target_var)
            jacobi_dot_x = T.Rop(p[T.arange(p.shape[0]), t], x, T.stack([-x[:,1],x[:,0]], axis=1))

            #[x_1*np.cos(angle)-x_2* np.sin(angle),
            # x_1*np.sin(angle)+x_2*np.cos(angle)]
            #[x_1* -sin(angle) - x_2*cos(angle),
            #        x_1*cos(angle)-x_2*sin(angle)]
            
            # 0 degrees:
            #[-x_2,
            # x_1]

            #180 degrees:
            #[x_2,
            #-x_1]
        else:
            jacobi_dot_x, _ = theano.scan(
                loop,
                non_sequences=[input_var, prediction, target_var],
                sequences=T.arange(input_var.shape[0])
            )

        return T.pow(jacobi_dot_x, 2).mean()
