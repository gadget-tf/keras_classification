from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
import sys

import re
import glob
from PIL import Image
import numpy as np

images = []
labels = []

def read_file(path):
    image = Image.open(path)
    image = np.asarray(image)
    return image

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
            print(dir)
            match = re.search(r"[0-9]+", dir)
            label = int(match.group())
            read_data(dir, label)
    else:
        dirs = glob.glob('./test/*')
        for dir in dirs:
            print(dir)
            match = re.search(r"[0-9]+", dir)
            label = int(match.group())
            read_data(dir, label)

    X_train = np.array(images)
    y_train = np.array(labels).reshape(-1, 1)

    return (X_train, y_train)

def img2data(fname):
    img = Image.open(fname)
    img = img.convert('RGB')
    img = img.resize((75, 75))
    data = np.asarray(img)
    data = np.reshape(data, (-1, 75 * 75 * 3))
    #data = data.reshape((-1, data_size))
    data = data / 256
    return data

def check(path, model):
    data = img2data(path)
    res = model.predict([data])[0]
    y = res.argmax()
    per = int(res[y] * 100)
    print('{0} ({1} %)'.format(labels[y], per))

def predict(isTest):
    (X_test, y_test)= make(isTest)

    X_test = X_test.astype('float32')
    X_test /= 255

    nb_classes = 3
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = load_model('model.h5')

    #check('./test/0-test/42183045.jpg', model)
    res = model.predict(X_test)

    acts = [np.argmax(i) for i in res]
    exps = [np.argmax(i) for i in Y_test]
    cnt = 0.0
    total = 0.0
    for (a,e) in zip(acts, exps):
        if a == e:
            cnt += 1.0
        total += 1.0

    print(cnt/total)

predict(0)
predict(1)