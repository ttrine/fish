from fish.detector_container import ModelContainer

from experiments.detect_batch_adam_64.train import construct

if __name__ == '__main__':
	import sys # basic arg parsing, infer ModelContainer name
	name = sys.argv[0].split('/')[-2]

	model = ModelContainer(name,construct(64),64,"sgd")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
