#!/usr/bin/python
from __future__ import print_function

import time

import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier


t0 = time.clock()
stopWords = text.ENGLISH_STOP_WORDS

def main():
	pathTrainingData = "testData/train.tsv"
	pathTestData = "testData/test.tsv"

	print("Importing '{:s}'...".format(pathTrainingData), end="")
	# Import training data
	train = pd.read_csv(pathTrainingData, 
						header=0, 
						delimiter="\t", 
						quoting=3,
						nrows = 1000	
						)
	printTime()

	numPhrasesTrain = train['Phrase'].size

	# Todo: Cleanup, stopwords...etc...

	# Create features with bag-of-words
	vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = stopWords, 
                             max_features = 5000,
                             lowercase = True)

	phrases = []
	sentiments = []
	for x in xrange(0,numPhrasesTrain):
		phrases.append(train['Phrase'][x])
		sentiments.append(train['Sentiment'][x])

	print("Fit and transform training features (phrases)", end="")
	train_data_features = vectorizer.fit_transform(phrases)
	printTime()

	train_data_features = train_data_features.toarray()

	print("Creating {:s}...".format("trained_data.txt"), end="")
   	trainedData = open("trained_data.txt", 'w')
   	trainedData.write(train_data_features)
   	printTime()

   	forest = RandomForestClassifier(n_estimators = 100, n_jobs = 4) 

   	train_data_labels = sentiments
   	
   	print("Fitting training data...", end="")
   	forest = forest.fit(train_data_features, train_data_labels)
   	printTime()



   	print("Importing '{:s}'...".format(pathTestData),end="")
	test = pd.read_csv(pathTestData, 
						header=0, 
						delimiter="\t", 
						quoting=3,
						nrows = 10)
	printTime()

   	numPhrasesTest = test['Phrase'].size

   	testPhrases = []
   	for x in xrange(0,numPhrasesTest):
		testPhrases.append(test['Phrase'][x])
	
	test_data = vectorizer.transform(testPhrases)
	test_data = test_data.toarray()

	print("Making prediction...", end="")
   	output = forest.predict(test_data)
   	printTime()

  	submissionFile = open("submission.csv", 'w')

  	submissionFile.write("PhraseID,Sentiment\n")

  	print("Creating submission file...", end="")
	for x in xrange(0,numPhrasesTest):
		submissionFile.write(str(test['PhraseId'][x])+","+str(output[x])+"\n")
	printTime()

def printTime():
	print("({:s})".format(str(time.clock()-t0)))
 	
if __name__=="__main__":
	print("Program has started!")
	main()
	print("Program has ended!", end="")
	printTime()