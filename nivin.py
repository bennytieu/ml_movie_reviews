import csv
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
#from findDuplicates import find_duplicates, replace_duplicates
#from replaceDuplicates import replace_duplicates,
#from HandleDuplicates import replace_duplicates, find_duplicates
from ProcessData import TrainingData, TestData
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC


def main():
	training_object = TrainingData("train.tsv") 
	train_sentiments = training_object.sentiments
	train_sentences = training_object.sentences

	test_object = TestData("test.tsv")
	test_sentences = test_object.sentences
	test_phraseId = test_object.phraseId
	
	text_clf = Pipeline([('vect', CountVectorizer(analyzer = "word",   \
		tokenizer = None,    \
		preprocessor = None, \
		#use a built-in stop word for english
		stop_words =  {text.ENGLISH_STOP_WORDS},   \
		#godtyckligt tal nu - satt en max = max antal ord?
		#True by default
		lowercase = True, \
		ngram_range = ( 1, 2), \
		#max_features = 20000,
		)),
		('tfidf', TfidfTransformer()), 
		('clf', LinearSVC(C = 0.3)), # Choose the classifier #MultinomialNB() LinearSVC() (train_sentences n_classes one-vs-rest classifiers), OneVsRestClassifier(LinearSVC())
		#('clf', RandomForestClassifier(n_estimators = 100, n_jobs = 4)), # Choose the classifier
		])

	phrases = []
	sentiments = []
	for x in xrange(0,len(train_sentences)):
		#lettersOnlyPhrase = re.sub("[\"('')]", "'", train['Phrase'][x])

		phrases.append(str(train_sentences[x]))
		sentiments.append(train_sentiments[x])

	text_clf = text_clf.fit(phrases, sentiments)

	phraseTest = []
	for x in xrange(0,len(test_sentences)):
		#lettersOnlyPhrase = re.sub("[^a-zA-Z]", " ", test['Phrase'][x])  
		phraseTest.append(str(test_sentences[x]))

	predictedData = text_clf.predict(phraseTest)

	with open("submission.csv", "w") as submissionFile:
		submissionFile.write("PhraseID,Sentiment\n")

		for x in xrange(0,len(test_sentences)):
			submissionFile.write(str(test_phraseId[x]) + "," + str(predictedData[x])+"\n")

main()
