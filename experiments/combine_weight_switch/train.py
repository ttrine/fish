import theano

from keras.engine.topology import Layer

from keras.models import Model

from keras.layers import Input, Dense, Flatten, merge, Reshape, BatchNormalization, ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.core import Masking, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.classify import ClassifierContainer

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

def construct(n):
	input_chunks = Input(shape=(None,n,n,3))
	chunks = BatchNormalization()(input_chunks)

	input_locations = Input(shape=(None,2))
	locations = BatchNormalization()(input_locations)

	# Glimpse net. Architecture inspired by DRAM paper.
	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(16, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((3, 3)))(chunks)
	chunks = TimeDistributed(Convolution2D(32, 5, 5, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	chunks = TimeDistributed(ZeroPadding2D((1, 1)))(chunks)
	chunks = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(chunks)
	chunks = TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(chunks)

	flattened_chunks = TimeDistributed(Flatten())(chunks)
	flattened_chunks = Masking()(flattened_chunks)
	feature_vectors = TimeDistributed(Dense(128,activation='relu'))(flattened_chunks)

	# Location encoder
	location_vectors = TimeDistributed(Dense(128,activation='relu'))(locations)
	location_vectors = Masking()(location_vectors)
	
	# Multiplicative where-what interaction
	hadamard_1 = merge([location_vectors, feature_vectors], mode='mul')

	# Combine the feature-location sequences and predict coverage sequence
	detect_rnn = LSTM(128, return_sequences=True, consume_less="gpu")(hadamard_1)
	detect_fcn = TimeDistributed(Dense(64,activation='relu'))(detect_rnn)
	cov_pr = TimeDistributed(Dense(1,activation='sigmoid'),name="coverage")(detect_fcn)

	# Prevent class gradient from flowing into coverage gradient
	cov_pr_nograd = TimeDistributed(GradientReversalLayer(0))(cov_pr)

	# Learn a scalar multiple of coverage probability for class inference
	cov_multiplier = TimeDistributed(Dense(1,activation='relu'))(cov_pr_nograd)

	# Combine the feature-location sequences
	cov_pr_repeated = TimeDistributed(RepeatVector(128))(cov_multiplier)
	cov_pr_repeated = TimeDistributed(Reshape((128,)))(cov_pr_repeated)
	hadamard_2 = merge([cov_pr_repeated, hadamard_1], mode='mul')

	# Scale by coverage probability and predict class
	classify_rnn = LSTM(128, consume_less="gpu")(hadamard_2)
	classify_fcn = Dense(64,activation='relu')(classify_rnn)
	class_pr = Dense(8,activation='softmax',name="class")(classify_fcn)

	return Model(input=[input_chunks,input_locations],output=[cov_pr,class_pr])

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(128),128,"adam",loss_weights=[1.,2.])
	model.model.load_weights('experiments/combine_weight/weights/44-1.9781.hdf5')
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
