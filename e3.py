# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model
from keras.optimizers import Adam, Adamax
from keras.utils import to_categorical

import os
import cv2

# Create a basis for all imported objects
PATH = 'DeepLearning/shapes/'
IMG_SIZE = 64 # Since 64x64 imgs
Shapes = ["circle", "square", "triangle"]

# Create holding arrays for images
Labels = []
Dataset = []

# Parse through each folder and pull all images
for shape in Shapes:
    print("Getting data for", shape)
    #iterate through each file in the folder
    for path in os.listdir(PATH + shape):
        #add the image to the list of images
        img = cv2.imread(PATH + shape + '/' + path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        Dataset.append(img)
        #add an integer to the labels list
        Labels.append(Shapes.index(shape))

# Print results
print("\nDataset Images size:", len(Dataset))
print("Image Shape:", Dataset[0].shape)
print("Labels size:", len(Labels))

Dataset = np.array(Dataset)
Dataset = Dataset.astype("float32") / 255.0

# One hot encode labels (preventing any integer relation from forming)
Labels = np.array(Labels)
Labels = to_categorical(Labels)

# Split Dataset to train\test
(trainX, testX, trainY, testY) = train_test_split(Dataset, Labels, test_size=0.2, random_state=42)

print("X Train shape:", trainX.shape)
print("X Test shape:", testX.shape)
print("Y Train shape:", trainY.shape)
print("Y Test shape:", testY.shape)

# Define the input layer to accept images
input_layer = Input((64,64,3))

# Flatten the input for dense processing
x = Flatten()(input_layer)

# Compute some dense layer processing
x = Dense(256, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(64, activation = 'relu')(x)

# Output layer as an integer value
output_layer = Dense(len(Shapes), activation = 'softmax')(x)

model = Model(input_layer, output_layer)

model.summary()

# Declare the optimizer
opt = Adamax(lr=0.0002)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
model.fit(trainX, trainY, batch_size=64, epochs=10, shuffle=True)

model.evaluate(testX, testY)

print('\n\nPredict Data')
Imgs = []
for path in os.listdir(PATH + 'Predict'):
    img = cv2.imread(PATH + 'Predict/' + path)
    print(PATH  + 'Predict/' + path, img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(img, 127, 255, 1)
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        cv2.drawContours(img,[cnt],0,(0,255,0),-1)

    img = cv2.merge([img,img,img])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print('***', img.shape)
    Imgs.append(img)

# Print results
print("\nDataset Images size:", len(Imgs))
print("Image Shape:", Imgs[0].shape)

Imgs = np.array(Imgs)
Imgs = Imgs.astype("float32") / 255.0

predictions = model.predict(Imgs)
print(predictions)

CLASSES = np.array(Shapes)
print(CLASSES[np.argmax(predictions, axis = 1)])
