#Premade 
import datetime
import os

#Custom
import submit


def main():
	classes = os.listdir("train")[1:]
	print classes
	for i in range(len(classes)):
		print len(os.listdir('train/' + classes[i]))
	
	#submit.sumbit_results()


main()