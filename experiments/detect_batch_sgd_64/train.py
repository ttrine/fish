from fish.detector import DetectorContainer

from experiments.detect_batch_adam_64.train import construct

if __name__ == '__main__':
	import sys # basic arg parsing, infer DetectorContainer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 4:
		print "Usage: train nb_epoch batch_size samples_per_epoch"
		sys.exit()

	model = DetectorContainer(name,construct(64),64,"sgd")
	model.train(nb_epoch=int(sys.argv[1]), batch_size=int(sys.argv[2]), samples_per_epoch=int(sys.argv[3]))
