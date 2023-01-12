from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

""" print(train_data.shape)
print(train_data[0])
print(train_data[0].shape) """

""" plt.imshow(train_data[0])
plt.show()
print(train_labels[0]) """

#create model
model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

#compile model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#model.summary()

# Change data to 2D & float
train_data = train_data.reshape((60000, 28*28))
train_data = train_data.astype('float32') / 255

test_data = test_data.reshape((10000, 28*28))
test_data = test_data.astype('float32') / 255

# Change labels to categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train model
model.fit(train_data, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_data, test_labels)
print('test_acc:', test_acc)















