
# coding: utf-8

# In[3]:


import csv
import cv2
import numpy as np
import sklearn
import scipy

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

def generator(lines, batch_size, augument):
    while True:
        for i in range(0, len(lines), batch_size):
            images = []
            measurements = []
            lines_batch = lines[i:i+batch_size]
            for line in lines_batch:
                correction = 0.23

                #center
                images.append(get_image(line[0]))
                measurements.append(float(line[3]))

                #left
                images.append(get_image(line[1]))
                measurements.append(float(line[3])+correction)

                #right
                images.append(get_image(line[2]))
                measurements.append(float(line[3])-correction)

            augument_images, augument_measurements = [], []
            if (augument):
                for image, measurement in zip(images, measurements):
                    augument_images.append(image)
                    augument_measurements.append(measurement)
                    augument_images.append(np.fliplr(image))
                    tmp = np.fliplr(image)
                    cv2.imwrite("./flip.jpg",tmp)
                    augument_measurements.append(measurement*-1)
                    X_train = np.array(augument_images)
                    y_train = np.array(augument_measurements)
            else:
                X_train = np.array(images)
                y_train = np.array(measurements)

            yield sklearn.utils.shuffle(X_train, y_train)

def get_image(path):
    filename = path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    image = scipy.misc.imresize(image, (80,160))
    image = image[35:70, :]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Lambda, Cropping2D, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

height = 35 
width = 160
top_remove = 25
bottom_remove = 10

model = Sequential()
model.add(Lambda(lambda x: (x/127.5)-1.0, input_shape=(height,width,3)))

model.add(Conv2D(24,5,5,activation='relu', border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(36,5,5,activation="relu", border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(48,5,5,activation="relu", border_mode='same', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64,3,3,activation="relu", border_mode='same', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(64,3,3,activation="relu", border_mode='same', subsample=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Dropout(0.5))

model.add(Flatten())

activation_full = 'tanh'

model.add(Dense(1164, activation=activation_full))
model.add(Dropout(0.5))

model.add(Dense(100, activation=activation_full))

model.add(Dense(50, activation=activation_full))

model.add(Dense(10, activation=activation_full))

model.add(Dense(1, activation=activation_full))

train_generator = generator(lines, 20, True)
validate_generator =  generator(lines, 20, False)

model.compile(loss="mae", optimizer=Adam(1e-4))
model.fit_generator(train_generator, samples_per_epoch=60000, nb_epoch=5, validation_data=validate_generator, nb_val_samples=17000, verbose=1)

model.save('model.h5')

