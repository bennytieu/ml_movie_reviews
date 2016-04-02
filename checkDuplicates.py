#!/usr/bin/python

import json
import pandas as pd 
import numpy as np

def main():
	with open("settings.json", "r") as jsonFile:
		jsonData = json.load(jsonFile)

		pathTrainingData = jsonData["train_path"]
		pathTestData = jsonData["test_path"]

		pdTrain = pd.read_csv(pathTrainingData, 
						header=0, 
						delimiter="\t", 
						quoting= jsonData["train_quoting"]
						)

		pdTest = pd.read_csv(pathTestData, 
							header=0, 
							delimiter="\t", 
							quoting= jsonData["test_quoting"],
							)

		npTrain = np.array(pdTrain['Phrase'])
		duplicateInTrain = []
		duplicateCount = 0
		with open("duplicateFile.csv", "w") as duplicateFile:
			duplicateFile.write("PhraseID,Sentiment,IndexInTest\n")
			for i in xrange(0,len(pdTest['Phrase'])):
				duplicateIndex = np.where(npTrain == pdTest['Phrase'][i])[0]

				if len(duplicateIndex) > 0:
					duplicateInTrain.append(duplicateIndex[0])
					duplicateCount += 1
					duplicateFile.write(str(pdTest['PhraseId'][i]) + "," + str(pdTrain['Sentiment'][duplicateIndex[0]]) + "," + str(i)+"\n")
					
		print "Found " + str(duplicateCount) + " duplicates"

if __name__=="__main__":
	print "Finding duplicates between test and training data"
	main()
	print "Finished"