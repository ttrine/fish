import csv
import numpy as np

def parse_losses(experiment, exp_type):
	path = "experiments/"+experiment+"/log.out"
	log = file(path,'r')
	lines = log.readlines()

	if exp_type == 'classifier':
		cov_train_losses = [float(line.split(' - loss: ')[-1].split('coverage_loss: ')[1].split(' - ')[0]) for line in lines if 'val_loss: ' in line]
		class_train_losses = [float(line.split(' - loss: ')[-1].split('class_loss: ')[1].split(' - ')[0]) for line in lines if 'val_loss: ' in line]
		
		cov_test_losses = [float(line.split(' - loss: ')[-1].split('val_coverage_loss: ')[1].split(' - ')[0]) for line in lines if 'val_loss: ' in line]
		class_test_losses = [float(line.split(' - loss: ')[-1].split('val_class_loss: ')[1].split(' - ')[0]) for line in lines if 'val_loss: ' in line]

		data = np.array(zip(cov_train_losses, class_train_losses, cov_test_losses, class_test_losses))

		# Dump to csv
		outpath = "analysis/training/"+experiment+".csv"
		f = file(outpath,'wb')
		w = csv.writer(f)
		w.writerow(['cov_train','class_train','cov_test','class_test'])
		w.writerows(data)
		f.close()
	else:
		cov_train_losses = [float(line.split(' - loss: ')[-1].split(' - ')[0]) for line in lines if 'val_loss' in line]
		cov_test_losses = [float(line.split(' - loss: ')[-1].split('val_loss: ')[-1].split('\n')[0]) for line in lines if 'val_loss' in line]
		
		data = np.array(zip(cov_train_losses, cov_test_losses))

		# Dump to csv
		outpath = "analysis/training/"+experiment+".csv"
		f = file(outpath,'wb')
		w = csv.writer(f)
		w.writerow(['cov_train','cov_test'])
		w.writerows(data)
		f.close()

if __name__ == '__main__':
	import sys # 1st arg is experiment name, second experiment type

	parse_losses(sys.argv[1], sys.argv[2])
