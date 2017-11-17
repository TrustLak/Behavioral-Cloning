# implement, train and validate the model
import csv 
import helpers
import tensorflow as tf
import pickle
import numpy as np
from sklearn.utils import shuffle
import cv2

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
###################################################################


# load data
###########
'''
load images and measurements.
correct measurements for left and right images.
augment the data by flipping the center, right, and left images.
'''
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#lines = []
new_data_index = len(lines) # pop this index form the new file\
print(new_data_index)

with open('train-data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines.pop(0) # remove first line
#lines.pop(new_data_index)
print(len(lines))

images = [] 
measurements = []
correction = 0.3
i = 0
print('Reading Data...')
for line in lines:
    if i%1000 == 0:
        print(i)
    if i > 1111000:
        print(i)
    # get left, center and right images:
    left_img, center_img, right_img = helpers.get_line_images(line, (i >= new_data_index - 1))

    # group grayscale of images:
    #image = helpers.combine_gray_images(left_img, center_img, right_img)
    #images.append(image)

    # augment and append data
    center_measurement = float(line[3])
    images.append(center_img)
    measurements.append(center_measurement)
    image_flipped = np.fliplr(center_img)
    measurement_flipped = - center_measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

    left_measurement = center_measurement + correction
    images.append(left_img)
    measurements.append(left_measurement)
    image_flipped = np.fliplr(left_img)
    measurement_flipped = - left_measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

    right_measurement = center_measurement - correction
    images.append(right_img)
    measurements.append(right_measurement)
    image_flipped = np.fliplr(right_img)
    measurement_flipped = - right_measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

    i += 1
print(i)
###################################################################

# Convert to numpy array
########################
X_train = np.array(images)
y_train = np.array(measurements)



# Implement the network architecture
####################################
# Build Convolutional Neural Network in Keras Here
# See sectoim 4 in nvidia paper: End to End learning for self-driving cars
# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
print('Building network...')
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape = (160, 320, 3)))
# add cropping to remove upper part of the image (useless for deciding steering angle)
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
###################################################################


# Split data and train the network
##################################
print('Training network...')
model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, nb_epoch=4, validation_split=0.2, shuffle=True)


# Save the network
##################
model.save('model.h5')
