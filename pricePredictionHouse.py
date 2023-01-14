import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.datasets import boston_housing
from keras import optimizers
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print(train_data.shape) # (404, 13)
print(train_labels[0]) # 15.2 in thousands of dollars

mean = train_data.mean(axis=0)
train_data = train_data - mean
std = train_data.std(axis=0)
train_data = train_data / std

test_data = test_data - mean
test_data = test_data / std

def buildModel(learning_Rate, input_Shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_Shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer= optimizers.RMSprop(learning_rate=learning_Rate), loss='mse', metrics=['mae'])
    return model

k = 4
num_ValSamples = len(train_data) // k
num_Epochs = 80
all_Scores = []

for i in range(k):
    print("Processing fold #", i)
    val_data = train_data[i * num_ValSamples: (i + 1) * num_ValSamples]
    val_targets = train_labels[i * num_ValSamples: (i + 1) * num_ValSamples]
    
    partial_train_data = np.concatenate(
        [train_data[:i * num_ValSamples],
        train_data[(i + 1) * num_ValSamples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_labels[:i * num_ValSamples],
        train_labels[(i + 1) * num_ValSamples:]],
        axis=0)
    
    model = buildModel(0.001, 13)
    history = model.fit(partial_train_data, partial_train_targets,
              epochs=num_Epochs,
              batch_size=16,
              validation_data=(val_data, val_targets),
              verbose=0)

    mae_history = history.history['val_mae']
    all_Scores.append(mae_history)
    print("Mean Absolute Error: ", mae_history[-1])

len(all_Scores[0]) # 80

average_Mae_History= pd.DataFrame(all_Scores).mean(axis=0)
print(average_Mae_History)

plt.plot(range(1, len(average_Mae_History[15:]) + 1), average_Mae_History[15:])
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

results = model.evaluate(test_data, test_labels)
print("Test Loss: ", results)


