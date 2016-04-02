# -*- coding: utf-8 -*-
import csv
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

class TrainingData:

	def __init__(self, train_file):

		self.train_file = train_file
	
		self.sentences = []
		self.sentiments = []
		self.data_features = [];
		#self.train_dict = dict()
		
		#return_tuple = collections.namedtuple('Return', ['train_data_features', 'train_sentiments', 'train_dict'])
		with open(train_file, "r") as f:
			#reads file and makes each row correspond to a list of strings
			#delimiter = "\t": the fields are seperated by tabs
			#quoting = 3: ignore doubled quotes
			train_data = csv.reader(f, delimiter = "\t", quoting = 3) 
			next(train_data, None)  # skip the headers
			for row in train_data:
				#row = clean_data(row)
				self.sentences.append(row[2])
				self.sentiments.append(row[3])
				
				'''			
				self.sentences.append(row[2])
				self.sentiments.append(row[3])
				'''
			'''
			self.data_features = vectorizer.fit_transform(self.sentences) # words before! Transforms the data to feature vectors
			self.data_features = self.data_features.toarray()
			'''
			#train_ret = return_tuple(train_data_features, train_sentiments, train_dict)
			print("--- Done vectorizing the train data ---")

			'''
			self.train_dict = dict(zip(self.train_sentences, self.train_sentiments))
			print("--- Done creating the dict from training data ---")
			'''
	
	def create_dict(self):
			# Make a dictrionary with sentences + sentiments
			train_dict = dict(zip(self.sentences, self.sentiments))
			print("--- Done creating the dict from training data ---")
			return train_dict


class TestData:

	def __init__(self, test_file):
		self.test_file = test_file

		self.sentences = []
		self.phraseId = []
		self.phraseId_index = []

		with open(test_file, "r") as f:
			test_data = csv.reader(f, delimiter = "\t", quoting = 3) 
			next(test_data, None)  # skip the headers

			index = 0;
			for row in test_data:
				self.phraseId.append(row[0])
				self.sentences.append(row[2])
				self.phraseId_index.append([row[0], index]) #Save index + phraseId - making it easy to access and merge
				index += 1;
			
		print("--- Done vectorizing the test data ---")

	def create_dict(self):
			# Make a dictrionary with sentences + sentiments
			test_dict = dict(zip(self.sentences, self.phraseId_index)) 
			print("--- Done creating the dict from test data ---")
			return test_dict

