#!/usr/bin/python
from __future__ import print_function

import time
import json
import sys
import re

import pandas as pd 
import numpy as np
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

startTime = time.time()
currentTime = startTime

def main():
	global currentTime
	exclude = set(string.punctuation)

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
										min_df = 0.000030,
										ngram_range=(1, 2)
										)),
							('tfidf', TfidfTransformer()),
							('lSVC', SVC(kernel='linear'))])	

		print("\t{: <40}".format("Fitting training data."), end="")
		sys.stdout.flush()

		phrases = []
		sentiments = []
		for x in xrange(0,len(train['Phrase'])):
			phrases.append(''.join(ch for ch in train['Phrase'][x] if ch not in exclude))

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
			phraseTest.append(''.join(ch for ch in test['Phrase'][x] if ch not in exclude))

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