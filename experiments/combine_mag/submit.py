from fish.classify import ClassifierContainer
from experiments.combine_mag.train import construct

def generate_submission(name, weight_file, clip):
	c = ClassifierContainer('combine_mag',construct(128),128,"adam")
	c.evaluate('74-1.0833.hdf5', clip=True)

if __name__ == '__main__':
	import sys # basic arg parsing, infer name
	name = sys.argv[0].split('/')[-2]
	
	if len(sys.argv) < 3:
		print "Usage: submit weight_file clip"
		sys.exit()

	c = ClassifierContainer(name,construct(128),128,"adam")
	c.evaluate(str(sys.argv[1]), bool(sys.argv[2]))
