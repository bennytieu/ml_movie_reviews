#!/usr/bin/python
from __future__ import print_function

import time
import json
import sys
import re

import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

startTime = time.time()
currentTime = startTime

def main():
	global currentTime

	with open("settings.json", "r") as jsonFile:
		jsonData = json.load(jsonFile)

		pathTrainingData = jsonData["train_path"]
		pathTestData = jsonData["test_path"]

		print("\t{: <40}".format("Importing training data."), end="")
		sys.stdout.flush()

		train = pd.read_csv(pathTrainingData, 
							header=0, 
							delimiter="\t",
							quoting = 3
							)
		printTime(currentTime)

		text_clf = Pipeline([('vect', CountVectorizer(
										analyzer = "word",
										tokenizer = None,
										preprocessor = None,
										stop_words = {text.ENGLISH_STOP_WORDS},
										lowercase = True,
										#max_df = 1.0,
										min_df = 0.000025,
										ngram_range=(1, 2)
										)),
							('tfidf', TfidfTransformer()),
							('lSVC', LinearSVC(C=0.4))])	

		print("\t{: <40}".format("Fitting training data."), end="")
		sys.stdout.flush()

		phrases = []
		sentiments = []
		for x in xrange(0,len(train['Phrase'])):
			#lettersOnlyPhrase = re.sub("[^a-zA-Z]", " ", train['Phrase'][x])  
			phrases.append(train['Phrase'][x])

		text_clf = text_clf.fit(phrases, train['Sentiment'])
		printTime(currentTime)

		print("\t{: <40}".format("Importing test data."), end="")
		sys.stdout.flush()

		test = pd.read_csv(	pathTestData,
							header=0,
							delimiter="\t",
							quoting = 3
							)
		printTime(currentTime)

		print("\t{: <40}".format("Making predictions for test."), end="")
		sys.stdout.flush()

		phraseTest = []
		for x in xrange(0,len(test['Phrase'])):
			#lettersOnlyPhrase = re.sub("[^a-zA-Z]", " ", test['Phrase'][x])  
			phraseTest.append(test['Phrase'][x])

		predictedData = text_clf.predict(phraseTest)
		printTime(currentTime)

		with open("submission.csv", "w") as submissionFile:
			submissionFile.write("PhraseID,Sentiment\n")

			print("\t{: <40}".format("Creating submission file"), end="")
			sys.stdout.flush()
			for x in xrange(0,len(test['Phrase'])):
				submissionFile.write(str(test['PhraseId'][x]) + "," + str(predictedData[x])+"\n")
			printTime(currentTime)

def printTime(t0):
	global currentTime
	currentTime = time.time()
	timeElasped = round(time.time()-t0,3)

	print("{: <50}".format(""+str(timeElasped)+" s"))
 	
if __name__=="__main__":
	print("Starting")
	main()
	print("Finished. Total time elapsed: ", end="")
	printTime(startTime)