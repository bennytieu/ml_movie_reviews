#!/usr/bin/python

import pandas as pd 

def main():
	duplicateFile = pd.read_csv("duplicateFile.csv", 
			header=0, 
			delimiter=",", 
			)

	submissionFile = pd.read_csv("submission.csv", 
			header=0, 
			delimiter=",", 
			)

	for x in xrange(0,len(duplicateFile['IndexInTest'])):
		dupPhraseID = duplicateFile['PhraseID'][x]
		dupSentiment = duplicateFile['Sentiment'][x]
		dupIndex = duplicateFile['IndexInTest'][x]

		submissionFile['PhraseID'][dupIndex] = dupPhraseID
		submissionFile['Sentiment'][dupIndex] = dupSentiment

	with open("submissionWithoutDuplicates.csv", "w") as submissionFileWithoutDuplicate:
			submissionFileWithoutDuplicate.write("PhraseID,Sentiment\n")

			for x in xrange(0,len(submissionFile['PhraseID'])):
				submissionFileWithoutDuplicate.write(str(submissionFile['PhraseID'][x])+","+str(submissionFile['Sentiment'][x])+"\n")

if __name__=="__main__":
	main()
