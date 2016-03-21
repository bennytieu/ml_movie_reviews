#!/usr/bin/python

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from StringIO import StringIO

def main():
    # Import training set to array [(phrase,sentiment)] 
    # Example: [("Jag gillar allt",5),("Jag hatar allt",1)]
    trainingSet = genfromtxt(
	    			open('testData/train.tsv','r') , 
	    			usecols=(2, 3), 
					delimiter='\t',
					dtype=None, 
					skip_header=1,
					# Some strings contains '#', don't know how to disable this
					comments='/nocomment/'
				)
    #print trainingSet

    # Phrase array
    # Example: [["Jag gillar allt",5],["Jag hatar allt",2]]
    train = [x[0] for x in trainingSet]
    print train

    encodedTrain = LabelEncoder().fit_transform(train)
    label = [x[1] for x in trainingSet]
    
    #encodedTrain = zip(encodedTrain,label)

    
    # Import test set to array [phraseID] 
    # Example: ["Hahaha","Hohoho"]
    testSet = genfromtxt(
				open('testData/test.tsv','r') , 
				usecols=(2),
				delimiter='\t',
				dtype=None,
				skip_header=1,
				comments='/nocomment/'
				)
    
    test = [x[1:] for x in testSet]

    encodedTestSet = LabelEncoder().fit_transform(test)
   

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    
    #rf.fit(encodedTrain,label)

    #pred = [[index + 1, x[1]] for index, x in enumerate(rf.predict(encodedTestSet))]
   
    #predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(encodedTestSet))]
    #print predicted_probs

    #savetxt('testData/submission2.csv', rf.predict(testSet), delimiter='\t', fmt='%s')
    
  

if __name__=="__main__":
    main()
    print "done"