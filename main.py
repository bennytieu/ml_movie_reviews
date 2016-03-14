#!/usr/bin/python

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
from StringIO import StringIO

def main():
    # Import training set to array [(phraseID,sentiment)] 
    # Example: [(1501,5),(1502,2)]
    trainingSet = genfromtxt(
	    			open('testData/train.tsv','r') , 
	    			usecols=(0, 3), 
					delimiter='\t',
					dtype=(int,int), 
					skip_header=1,
					# Some strings contains '#', don't know how to disable this
					comments='/nocomment/'
				)

    # Phrase array
    # Example: [[1501,5],[1502,2]]
    train = [x[0:] for x in trainingSet]
    #print train

	# Sentiment array
	# Example: [5,2]    
    target = [x[1] for x in trainingSet]
    #print target

    # Import test set to array [phraseID] 
    # Example: [1501,1502]
    testSet = genfromtxt(
				open('testData/test.tsv','r') , 
				usecols=(0,2),
				delimiter='\t',
				dtype=(int),
				skip_header=1,
				comments='/nocomment/'
				)
    
    #print testSet

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    
    rf.fit(train,target)

    print rf.predict(testSet)

    #savetxt('testData/submission2.csv', rf.predict(testSet), delimiter='\t', fmt='%s')

if __name__=="__main__":
    main()
    print "done"