import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#print(train_data.shape) # (8982,)
#print(train_labels[0]) # 3 of 46 categories

wordIndex = reuters.get_word_index()

reverseWordIndex = dict([(value, key) for (key, value) in wordIndex.items()])

decodedReview = ' '.join([reverseWordIndex.get(i - 3, '-') for i in train_data[0]])

def vectorizeSequence(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorizeSequence(train_data)
x_test = vectorizeSequence(test_data)

y_train = to_categorical(train_labels) # array([0.,0.,1.,...], dtype=float32)
y_test = to_categorical(test_labels) 

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10000,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(
  partial_x_train, partial_y_train,
  epochs=9,
  batch_size=512,
  validation_data=(x_val, y_val))

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

#fig = plt.figure(figsize=(10, 5))
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, acc_values, 'o', label='Training')
plt.plot(epochs, val_acc_values, '--', label='Validation')
plt.title('Training and validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results) # 78% [0.9447727203369141, 0.7876224517822266]

predictions = model.predict(x_test)
print(predictions[0].shape) # (46,)
print(np.sum(predictions[0])) # 1.0

print(np.argmax(predictions[0])) # 3









  

