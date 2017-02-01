import csv

def parse_losses(experiment):
	path = "experiments/"+experiment+"/log.out"
	log = file(path,'r')
	lines = log.readlines()
	val_losses = [[float(line.split('val_loss: ')[-1].split('\n')[0])] for line in lines if 'val_loss: ' in line]

	# Dump to 1 column "csv"
	outpath = "analysis/training/"+experiment+".csv"
	f = file(outpath,'wb')
	w = csv.writer(f)
	w.writerows(val_losses)
	f.close()

if __name__ == '__main__':
	import sys # 1st arg is experiment name

	parse_losses(sys.argv[1])
