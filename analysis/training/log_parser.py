## Rudimentarily parses out validation losses
## from a log. Assumes that first line of file is Epoch 1.
import csv

def parse_losses(logpath):
	log = file(logpath,'r')
	lines = log.readlines()
	train_lines = [line for ind,line in enumerate(lines) if ind%2==1]
	val_losses = [float(train_line.split(' ')[-1]) for train_line in train_lines]
	as_rows = [[val_loss] for val_loss in val_losses]

	# Dump to 1 column "csv"
	losspath = logpath.split('.')[0]+".csv"
	lossfile = file(losspath,'wb')
	losswriter = csv.writer(lossfile)
	losswriter.writerows(as_rows)
	lossfile.close()

if __name__ == '__main__':
	import sys
	parse_losses(sys.argv[1]) # Assumes 1st arg is filepath