import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import imdb
import matplotlib.pyplot as plt

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

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(
 partial_x_train, partial_y_train,
 epochs=4,
 batch_size=512,
 validation_data=(x_val, y_val))

history_dict = history.history
#print(history_dict)

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'o', label='Training loss')
plt.plot(epochs, val_loss_values, '--', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results) # [0.2896219491958618, 0.883400022983551]

 


