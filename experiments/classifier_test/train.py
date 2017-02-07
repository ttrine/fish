from keras.models import Model
from keras.layers import Input, Dense, Flatten, merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM

from fish.classifier import ClassifierContainer

def construct(n):
	chunk_input = Input(shape=(None,n,n,3))
	location_input = Input(shape=(None,2))

	chunk_flat = TimeDistributed(Flatten())(chunk_input)
	feature_vector = TimeDistributed(Dense(10,activation='softmax'))(chunk_flat)

	location_vector = TimeDistributed(Dense(10,activation='softmax'))(location_input)

	model = merge([location_vector, feature_vector], mode='mul')
	model = LSTM(10)(model)
	model = Dense(8,activation='softmax')(model)

	return Model(input=[chunk_input,location_input],output=model)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = ClassifierContainer(name,construct(256),256,"adam")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
