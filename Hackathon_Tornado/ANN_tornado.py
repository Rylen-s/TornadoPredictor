import tensorflow as tf
from tensorflow import keras
from keras import layers, models, datasets
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("./Dataset/tornado_dataset.csv")
df['Tornado'].replace({'Yes': 1, 'No': 0}, inplace=True)

X = df.drop('Tornado', axis='columns')
y = df['Tornado']

X_Train, X_test, y_train, y_test = train_test_split(X, y, test_size=(0.2), random_state=5)

model = keras.Sequential([
    keras.layers.Dense(6, input_shape=(5,), activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_Train, y_train, epochs = 100)
#Evaluating CNN
model.evaluate(X_test,y_test)

#Printing the classification report
y_pred = model.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification report: \n", classification_report(y_test,y_pred_classes))

#Pickling the trained CNN model
model.save('Tornado_Predictor_87.h5')

