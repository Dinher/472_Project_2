import os, re, json, operator
from math import log10 as log

list_training_files = os.listdir('train')
list_testing_files = os.listdir('test')
classes=['ham','spam']
ham_tokens = {}
spam_tokens = {}
prob = {'ham':0, 'spam':0}
count = {'ham':0, 'spam':0, 'total_docs':0}

def tokenizeFreq(file_list, token_dict, email_type):
	for file_name in file_list:
		with open('train/'+file_name,'r',encoding='latin-1') as file:
			if email_type in file_name:
				count[email_type] += 1
				count['total_docs'] += 1
				raw = file.read()
				fold = raw.lower()
				tokenized = re.split(r'[^a-zA-Z]',fold)
				tokenized = [i for i in tokenized if i.strip()]
				
				for i in tokenized:
					if i in token_dict:
						token_dict[i][email_type+'_frequency'] += 1
					else:
						word = {'ham_frequency':  0, 'spam_frequency': 0, 'spam_cond': 0, 'ham_cond':0}
						word[email_type+'_frequency'] += 1
						token_dict[i] = word

def getCondProb(token_dict):
	delta = 0.5
	for c in classes:
		for i in token_dict:
			freq = token_dict[i][ c + '_frequency']
			token_dict[i][ c + '_cond'] = ( (freq+delta) / (count[c] + (count['total_docs'] * delta)) ) 
	
def toFile(dictionary):
	with open('model.txt', 'w') as file:
		sorted_dict = list(dictionary.keys())
		sorted_dict.sort()
		for item in sorted_dict:
			freq = ' ' + str(dictionary[item]['ham_frequency']) + ' ' + str(dictionary[item]['spam_frequency'])
			cond = ' ' + str(dictionary[item]['ham_cond']) + ' ' + str(dictionary[item]['spam_cond'])
			file.write(item + freq + cond + '\n')

def combineTokens(d1,d2):
	z = json.loads(json.dumps(d1))
	for item in d2:
		if item in z:
			for key in z[item].keys():
				if z[item][key] < d2[item][key]:
					z[item][key] = d2[item][key]
		else:
			z[item] = d2[item]
	return z

def classifyTraining(a):
	total_docs = count['ham'] + count['spam']
	for c in classes:
		prob[c] = count[c] / total_docs
	print('PROBABILITIES')
	print(prob)

def getScore(doc, z):
	score = {}
	tokenized_doc = re.split(r'[^a-zA-Z]',doc)
	tokenized_doc = [i for i in tokenized_doc if i.strip()]
	for c in classes:
		score[c] = 0
		score[c] = prob[c]
		for word in tokenized_doc:
			if word in z:			
				score[c] = score[c] * (z[word][c + '_cond'])
	#print(score)
	classified = max(score.items(), key=operator.itemgetter(1))[0]
	return classified, score['ham'], score['spam']

def classifyTests():
	class_errors = 0
	to_file = ''
	for file in list_testing_files:
		line_count = 1
		with open('test/'+file,'r', encoding='latin-1') as f:
			email_type = 'ham'
			if 'spam' in file:
				email_type = 'spam'
			document = f.read()
			score = getScore(document,z)
			error = 'right'
			if email_type != score[0]:
				error = 'wrong'
				class_errors += 1
			
			to_file += str(line_count) + '  ' + file + '  ' + score[0] + '  ' + str(score[1]) + '  ' + str(score[2]) + '  ' + email_type + '  ' + error + '\n'
			line_count += 1
	
	with open('baseline-result.txt', 'a') as bf:
		bf.write(to_file)
	
	return class_errors


tokenizeFreq(list_training_files, ham_tokens, 'ham')
tokenizeFreq(list_training_files, spam_tokens, 'spam')

print(len(ham_tokens))
print(len(spam_tokens))

z = combineTokens(ham_tokens, spam_tokens)

print(len(z))
getCondProb(z)
toFile(z)

classifyTraining(z)

num_errors = classifyTests()
print('Wrong classifications: ' + str(num_errors))
