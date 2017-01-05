def get_splits():
	import csv

	has_fish = []
	no_fish = []

	with open('data/models/output.csv','r') as preds:
		predreader = csv.reader(preds)
		predreader.next()
		for pred in predreader:
			if float(pred[1])>float(pred[2]):
				has_fish.append(pred[0])
			else:
				no_fish.append(pred[0])
	return (has_fish,no_fish)
