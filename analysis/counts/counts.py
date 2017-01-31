import csv

from fish.detector_container import read_boxes

y_boxes = read_boxes()

lens = [[len(y_boxes[i])] for i in y_boxes]

f = file("analysis/fish_count/counts.csv",'wb')
w = csv.writer(f)
w.writerows(lens)
f.close()
