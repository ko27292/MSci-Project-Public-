import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm

def gen_data(train_video, test_video):
    cap = CapMultiThreading(train_video)
    cap2 = CapMultiThreading(test_video)


    frames = []
    for i in tqdm(range(cap.get_frame_count())):
        ret, frame = cap.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #frame = frame - np.median(frame)
        #frame = frame/np.max(frame)
        frames.append(frame/255.0)
    
    train_images = np.stack(frames, axis=0)

    frames = []
    for i in tqdm(range(cap2.get_frame_count())):
        ret, frame = cap2.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #frame = frame - np.median(frame)
        #frame = frame/np.max(frame)
        frames.append(frame/255.0)

    test_images = np.stack(frames, axis=0)


    return train_images, test_images


def train_CNN(train_images, train_labels):

    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, 5, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.Conv2D(128, 3, activation='relu'))
    model.add(keras.layers.Conv2D(4, 1, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(2))

    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

    model.fit(train_images[:8000], train_labels[:8000], epochs=15, validation_split=0.2, shuffle=True )
    model.evaluate(train_images[8000:9990], train_labels[8000:9990])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    print(train_images[-1].shape)
    pred = model.predict(test, batch_size=1)

    x, y = pred[0]

    print(x, y)
    plt.figure()
    plt.imshow(train_images[-1])
    plt.plot(x, y, "ro")
    plt.show()
    
def dense(train_images, train_labels):

    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    print(input_shape)
    print(train_images.shape)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(2))

    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse'])

    model.fit(train_images[:8000], train_labels[:8000], epochs=30, validation_split=0.2, shuffle=True)

    model.evaluate(train_images[8000:9990], train_labels[8000:9990])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    x, y = pred[0]

    print(x, y, train_labels[-1])
    plt.figure()
    plt.imshow(train_images[-1])
    plt.plot(x, y, "ro")
    plt.show()

    model.save("dense.keras")

def reload_dense(train_images, train_labels):
    model = keras.models.load_model("dense.keras")

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    x, y = pred[0]

    print(x, y, train_labels[-1])
    plt.figure()
    plt.imshow(train_images[-1])
    plt.plot(x, y, "ro")
    plt.show()

def train_z(train_images, train_labels):
    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(1))

    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse'])

    model.fit(train_images[:700], train_labels[:700], epochs=30, validation_split=0.2, shuffle=True)

    model.evaluate(train_images[700:705], train_labels[700:705])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    z = pred[0]

    print(z, train_labels[-1])
    

    model.save("z_model.h5")
