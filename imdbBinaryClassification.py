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
 epochs=20,
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

# Less complex model variant
model2 = Sequential()
model2.add(Dense(4, activation='relu', input_shape=(10000,)))
model2.add(Dense(4, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history2 = model2.fit(
    partial_x_train, partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

history_dict2 = history2.history

loss_values2 = history_dict2['loss']
val_loss_values = history_dict['val_loss']
val_loss_values2 = history_dict2['val_loss']

epochs = range(1, len(loss_values2) + 1)

plt.plot(epochs, val_loss_values2, 'o', label='Smaller model')
plt.plot(epochs, val_loss_values, '--', label='Original model')
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 


# Regularization model

from keras import regularizers

model3 = Sequential()
model3.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
model3.add(Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history3 = model3.fit(
    partial_x_train, partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

history_dict3 = history3.history

loss_values3 = history_dict3['loss']
val_loss_values3 = history_dict3['val_loss']

epochs = range(1, len(loss_values3) + 1)

plt.plot(epochs, val_loss_values3, 'o', label='Regularized model')
plt.plot(epochs, val_loss_values, '--', label='Original model')
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Dropout model
from keras.layers import Dropout

model4 = Sequential()
model4.add(Dense(4, activation='relu', input_shape=(10000,)))
model4.add(Dropout(0.5))
model4.add(Dense(4, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(1, activation='sigmoid'))

model4.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history4 = model4.fit(
    partial_x_train, partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

history_dict4 = history4.history

loss_values4 = history_dict4['loss']
val_loss_values4 = history_dict4['val_loss']

epochs = range(1, len(loss_values4) + 1)

plt.plot(epochs, val_loss_values4, 'o', label='Dropout model')
plt.plot(epochs, val_loss_values, '--', label='Original model')
plt.title('Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


