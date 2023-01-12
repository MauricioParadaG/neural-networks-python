import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data.shape) # (25000,)
#print(train_data[0]) # [1, 14, ...]

print(train_labels[0]) # 1

wordIndex = imdb.get_word_index()
#print(wordIndex) # {'paget': 18509, 'expands': 20597}
reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

decodedReview = ' '.join([reverseWordIndex.get(i - 3, '-') for i in train_data[0]]) 
#print(decodedReview)

def vectorizeSequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorizeSequence(train_data) 
x_test = vectorizeSequence(test_data) 

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


