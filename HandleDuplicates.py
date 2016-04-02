import csv

# Nbr of dupicates (on all data) = 10297
def find_duplicates(file, test_dict, train_dict):

	#create a new file with the format: index, phraseId, sentiment
	with open(file, "w") as duplicates:
		csvwriter = csv.writer(duplicates, delimiter=",")
		csvwriter.writerow(["Index PhraseId Sentiment"])
		for key_sentence, praseId_index in test_dict.items() :
			sentiment = train_dict.get(key_sentence)
			if (sentiment):
				#print (str(praseId_index[1]) + ", " + praseId_index[0] + ", " + sentiment) #index, phraseId, sentiment
				csvwriter.writerow([str(praseId_index[1]) + " " + praseId_index[0] + "," + sentiment])
	print("klar med duplicates")

	#return result;


def replace_duplicates(dup_file_, sub_file_):
	
	#Open submission file and save all data in an array
	sub_data_array = []
	with open(sub_file_, "r") as sub_file:
		sub_file.readline() #Skip the header
		for line in sub_file:
			#print (line)
			line = line.strip("\n")
			sub_data_array.append(line)


	#Open duplicate file and replace all data in the sub_array with data from the duplicate file
	with open(dup_file_, "r") as dup_file:
		dup_file.readline() #Skip the header
		for line in dup_file:
			line = line.split()
			index = int(line[0].strip('"')); # Strip ovan tog bara bort the first '"'
			sentence = line[1].strip('"');
			#print ("BEFORE")
			#print(sub_data_array[index])
			sub_data_array[index] = sentence
			#print ("AFTER")
			#print(sub_data_array[index])

	#Overwrite the dub_file with the new (merged) data
	with open(sub_file_,'w') as f:
		f.write("PhraseID,Sentiment\n")
		for item in sub_data_array:
  			f.write("%s\n" % item)


# put findDuplicates here