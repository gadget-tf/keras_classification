# coding: utf-8

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

import re
import numpy as np
import glob
from PIL import Image

from keras.callbacks import CSVLogger
csv_logger = CSVLogger('log.csv', append=True, separator=';')

classes = 0
data_size = 75 * 75 * 3
batch_size = 32
epoch_size = 10

images = []
labels = []

def read_data(path, label):
    files = glob.glob(path + '/*.jpg')
    for file in files:
        image = Image.open(file)
        image = np.asarray(image)
        images.append(image)
        labels.append(label)

def make(isTest):
    global images
    global labels
    images = []
    labels = []

    if isTest == 0:
        dirs = glob.glob('./train/*')
        for dir in dirs:
            match = re.search(r"[0-9]+", dir)
            label = int(match.group())
            read_data(dir, label)
    else:
        dirs = glob.glob('./test/*')
        for dir in dirs:
            match = re.search(r"[0-9]+", dir)
            label = int(match.group())
            read_data(dir, label)

    X_train = np.array(images)
    y_train = np.array(labels).reshape(-1, 1)

    return (X_train, y_train)

def train(X_train, y_train, X_test, y_test):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(X_train)

    model.summary()

    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train),
                        epochs=epoch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[csv_logger])

    model.save('model.h5')

def main():
    global classes
    classes = len(glob.glob('./train/*'))
    print(classes)

    X_train, y_train = make(0)
    X_test, y_test = make(1)

    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()