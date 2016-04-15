#!/usr/bin/python
from __future__ import print_function

import time
import sys
import re

import pandas as pd 
import numpy as np
import string
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsOneClassifier
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer

startTime = time.time()
currentTime = startTime



def main():
	global currentTime

	pathTrainingData = "data/train.tsv"
	pathTestData = "data/test.tsv"

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
									tokenizer = tokenize,
									preprocessor = None,
									#stop_words = {text.ENGLISH_STOP_WORDS},
									lowercase = True,
									max_df = 1.0,
									min_df = 0.000040,
									ngram_range=(1, 2)
									)),
						('tfidf', TfidfTransformer()),
						('lSVC', OneVsOneClassifier(LinearSVC(C=0.32)))])	

	print("\t{: <40}".format("Fitting training data."), end="")
	sys.stdout.flush()

	phrases = []
	for x in xrange(0,len(train['Phrase'])):
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

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    res = stemTokens(tokens)
    #res = lemLokens(tokens)

    return res

def stemTokens(tokens):
	stemmer = PorterStemmer()
	stemmed = []
	for token in tokens:
		stemmed.append(stemmer.stem(token))
	return stemmed

def lemLokens(tokens):
	lemmer = nltk.stem.WordNetLemmatizer()
	lemmed = []
	for token in tokens:
		stemmed.append(lemmer.lemmatize(token))
	return lemmed

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