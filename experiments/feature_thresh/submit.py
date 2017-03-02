import numpy as np

from fish.classify import ClassifierContainer
from .train import *

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 3:
		print "Usage: submit weight_file clip"
		sys.exit()
	
	c = ClassifierContainer(name,construct(),32,"adam")
	c.model.load_weights('experiments/'+name'+/weights/'+str(sys.argv[1]))

	# Load the fish feature mask. Just derive from the last set of weights.
	filter_weights = np.array(c.model.layers[-4].weights[0].eval()).reshape(256)
	inference_fish_feats = (filter_weights > 0).astype(np.float32)
	K.set_value(fish_feats, inference_fish_feats)
	
	c.evaluate(str(sys.argv[1]), bool(sys.argv[2]))
