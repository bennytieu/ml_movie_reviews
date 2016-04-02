#!/usr/bin/python
from __future__ import print_function

import time
import json
import sys

import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import GaussianNB


startTime = time.clock()
currentTime = startTime

def main():
	global currentTime

	with open("settings.json", "r") as jsonFile:
		jsonData = json.load(jsonFile)

		pathTrainingData = jsonData["train_path"]
		pathTestData = jsonData["test_path"]

		# Import training data
		print("\t{: <40}".format("Importing training data."), end="")
		sys.stdout.flush()

		if jsonData["train_n_rows"] == -1: jsonTrainNRows = None
		else: jsonTrainNRows = jsonData["train_n_rows"]

		train = pd.read_csv(pathTrainingData, 
							header=0, 
							delimiter="\t", 
							quoting= jsonData["train_quoting"],
							nrows = jsonTrainNRows
							)
		printTime(currentTime)



		if(jsonData["stop_words"]): stopWords = text.ENGLISH_STOP_WORDS
		else: stopWords = None
		vectorizer = CountVectorizer(	analyzer = "word",
										tokenizer = None,
										preprocessor = None,
										stop_words = stopWords, 
										max_features = jsonData["max_features"],
										lowercase = jsonData["lowercase"],
										ngram_range=(1, 1)
										)

		phrases = []
		sentiments = []
		numPhrasesTrain = train['Phrase'].size
		for x in xrange(0,numPhrasesTrain):
			phrases.append(train['Phrase'][x])
			sentiments.append(train['Sentiment'][x])

		print("\t{: <40}".format("Fit and transform training features."), end="")
		sys.stdout.flush()

		traningDataFeatures = vectorizer.fit_transform(phrases)
		printTime(currentTime)

		traningDataFeatures = traningDataFeatures.toarray()

		randomForest = GaussianNB()

		trainingDataLabels = sentiments
		
		print("\t{: <40}".format("Fitting training data."), end="")
		sys.stdout.flush()
		randomForest = randomForest.fit(traningDataFeatures, trainingDataLabels)
		printTime(currentTime)

		print("\t{: <40}".format("Importing test data."), end="")
		sys.stdout.flush()

		if jsonData["test_n_rows"] == -1: jsonTestNRows = None
		else: jsonTestNRows = jsonData["test_n_rows"]
		test = pd.read_csv(	pathTestData, 
							header=0, 
							delimiter="\t", 
							quoting=jsonData["train_quoting"],
							nrows = jsonTestNRows
							)
		printTime(currentTime)

		numPhrasesTest = test['Phrase'].size

		testPhrases = []
		for x in xrange(0,numPhrasesTest):
			testPhrases.append(test['Phrase'][x])

		print("\t{: <40}".format("Transform test phrases."), end="")
		testData = vectorizer.transform(testPhrases)
		testData = testData.toarray()
		printTime(currentTime)

		print("\t{: <40}".format("Making predictions for test."), end="")
		sys.stdout.flush()
		predictedData = randomForest.predict(testData)
		printTime(currentTime)

		with open("submission.csv", "w") as submissionFile:
			submissionFile.write("PhraseID,Sentiment\n")

			print("\t{: <40}".format("Creating submission file"), end="")
			sys.stdout.flush()
			for x in xrange(0,numPhrasesTest):
				submissionFile.write(str(test['PhraseId'][x])+","+str(predictedData[x])+"\n")
			printTime(currentTime)

def printTime(t0):
	global currentTime
	currentTime = time.clock()
	timeElasped = round(time.clock()-t0,3)

	print("{: <50}".format(""+str(timeElasped)+" s"))
 	
if __name__=="__main__":
	print("Starting...")
	main()
	print("End. Total time elapsed: ", end="")
	printTime(startTime)