import theano

from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K

import theano.tensor as T

class ReverseGradient(theano.Op):
    """ theano operation to reverse the gradients
    Introduced in http://arxiv.org/pdf/1409.7495.pdf
    """

    view_map = {0: [0]}

    __props__ = ('hp_lambda', )

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        assert hasattr(self, '_props'), "Your version of theano is too old to support __props__."
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

class GradientReversalLayer(Layer):
    """ Reverse a gradient 
    <feedforward> return input x
    <backward> return -lambda * delta
    """
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda
        self.gr_op = ReverseGradient(self.hp_lambda)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return self.gr_op(x)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {"name": self.__class__.__name__,
                         "lambda": self.hp_lambda}
        base_config = super(GradientReversalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SpecialBatchNormalization(Layer):
    # Normalizes nonblack elements of the input only.
    ## Designed for use as input to the first layer of a model
    ## whose images are of varying size and placed in a padded tensor.
    def __init__(self, epsilon=1e-5, mode=0, axis=-1, momentum=0.99,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
        self.supports_masking = True
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        if self.mode == 0:
            self.uses_learning_phase = True
        super(SpecialBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.regularizers = []
        if self.gamma_regularizer:
            self.gamma_regularizer.set_param(self.gamma)
            self.regularizers.append(self.gamma_regularizer)

        if self.beta_regularizer:
            self.beta_regularizer.set_param(self.beta)
            self.regularizers.append(self.beta_regularizer)

        self.running_mean = K.zeros(shape,
                                    name='{}_running_mean'.format(self.name))
        self.running_std = K.ones(shape,
                                  name='{}_running_std'.format(self.name))
        self.non_trainable_weights = [self.running_mean, self.running_std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        self.called_with = None

    def call(self, x, mask=None):
        assert self.built, 'Layer must be built before being called'
        input_shape = self.input_spec[0].shape

        reduction_axes = [0]
        broadcast_shape = [1] * (len(input_shape) - 1)
        broadcast_shape[self.axis] = input_shape[self.axis]

        self.called_with = x

        # Remove padded elements before computing statistics
        x_nonempty = x.nonzero_values()

        x_normed, mean, std = K.normalize_batch_in_training(
            x_nonempty, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        self.updates = [K.moving_average_update(self.running_mean, mean, self.momentum),
                        K.moving_average_update(self.running_std, std, self.momentum)]

        x_normed_running = K.batch_normalization(
            x_nonempty, self.running_mean, self.running_std,
            self.beta, self.gamma,
            epsilon=self.epsilon)

        # pick the normalized form of x corresponding to the training phase
        x_normed = K.in_train_phase(x_normed, x_normed_running)
        x_normed_masked = T.set_subtensor(x[x.nonzero()], x_normed)

        return x_normed_masked

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  "mode": self.mode,
                  "axis": self.axis,
                  "gamma_regularizer": self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  "beta_regularizer": self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  "momentum": self.momentum}
        base_config = super(SequenceBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SequenceBatchNormalization(Layer):
    # Implements sequence-wise normalization of feature maps. 
    ## Based heavily upon BatchNormalization layer code.
    def __init__(self, epsilon=1e-5, mode=0, axis=-1, momentum=0.99,
                 weights=None, beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
        self.supports_masking = True
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        if self.mode == 0:
            self.uses_learning_phase = True
        super(SequenceBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.regularizers = []
        if self.gamma_regularizer:
            self.gamma_regularizer.set_param(self.gamma)
            self.regularizers.append(self.gamma_regularizer)

        if self.beta_regularizer:
            self.beta_regularizer.set_param(self.beta)
            self.regularizers.append(self.beta_regularizer)

        self.running_mean = K.zeros(shape,
                                    name='{}_running_mean'.format(self.name))
        self.running_std = K.ones(shape,
                                  name='{}_running_std'.format(self.name))
        self.non_trainable_weights = [self.running_mean, self.running_std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        self.called_with = None

    def call(self, x, mask=None):
        if self.mode == 0 or self.mode == 2:
            assert self.built, 'Layer must be built before being called'
            input_shape = self.input_spec[0].shape

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            del reduction_axes[-1]
            broadcast_shape = [1] * (len(input_shape) - 1)
            broadcast_shape[self.axis] = input_shape[self.axis]
            if self.mode == 2:
                x_normed, mean, std = K.normalize_batch_in_training(
                    x, self.gamma, self.beta, reduction_axes,
                    epsilon=self.epsilon)
            else:
                # mode 0
                if self.called_with not in {None, x}:
                    raise Exception('You are attempting to share a '
                                    'same `SequenceBatchNormalization` layer across '
                                    'different data flows. '
                                    'This is not possible. '
                                    'You should use `mode=2` in '
                                    '`SequenceBatchNormalization`, which has '
                                    'a similar behavior but is shareable '
                                    '(see docs for a description of '
                                    'the behavior).')
                self.called_with = x

                # Remove padded sequence elements before computing statistics
                x_nonempty = x[T.any(x, range(2,x.ndim)).nonzero()]

                x_normed, mean, std = K.normalize_batch_in_training(
                    x_nonempty, self.gamma, self.beta, reduction_axes,
                    epsilon=self.epsilon)

                self.updates = [K.moving_average_update(self.running_mean, mean, self.momentum),
                                K.moving_average_update(self.running_std, std, self.momentum)]

                if K.backend() == 'tensorflow' and sorted(reduction_axes) == range(K.ndim(x))[:-1]:
                    x_normed_running = K.batch_normalization(
                        x_nonempty, self.running_mean, self.running_std,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        x_nonempty, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                x_normed = K.in_train_phase(x_normed, x_normed_running)
                x_normed_masked = T.set_subtensor(x[T.any(x, range(2,x.ndim)).nonzero()], x_normed)

        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(x, axis=-1, keepdims=True)
            std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (x - m) / (std + self.epsilon)
            x_normed = self.gamma * x_normed + self.beta
        return x_normed_masked

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  "mode": self.mode,
                  "axis": self.axis,
                  "gamma_regularizer": self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  "beta_regularizer": self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  "momentum": self.momentum}
        base_config = super(SequenceBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
