import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf


from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm
from tensorflow import keras
from scipy import linalg
from sklearn.decomposition import PCA
from keras import Input

def gen_canny(file_name):
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    frames = []
    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 50, 200)
        frames.append(edges)

    train_images = np.stack(frames, axis=0)

    return train_images

def get_PCA(data, N=100):
    '''A function to get the PCA components of a given data set
    Inputs:
        data - the data to be used
    Outputs:
        vecs - eigenvectors
        vals - eigenvalues
        x - x values
        mu - mean of the data
    '''
    X = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
    mu = np.mean(X, axis=0)
    x = X - mu
    rho = np.cov(x, rowvar=False)
    vals, vecs = linalg.eigh(rho)
    vecs = np.flip(vecs)
    vals = np.flip(vals)

    P_train = np.dot(x, vecs)
    new_train = (np.dot(P_train[:,0:N], vecs.T[0:N,:])) + mu
    new_train = new_train.reshape(data.shape[0],data.shape[1],data.shape[2])
    return new_train

def sk_pca(data, N):
    red = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
    pca = PCA(N)
    return pca.fit_transform(red), pca

def pca_single(image, N=100):
    '''A function to get the PCA components of a given data set
    Inputs:
        data - the data to be used
    Outputs:
        vecs - eigenvectors
        vals - eigenvalues
        x - x values
        mu - mean of the data
    '''
    image = np.expand_dims(image, axis=0)
    X = np.squeeze(image)
    mu = np.mean(X, axis=0)
    x = X - mu
    rho = np.cov(x, rowvar=False)
    vals, vecs = linalg.eigh(rho)
    vecs = np.flip(vecs)
    vals = np.flip(vals)

    P_train = np.dot(x, vecs)
    new_train = (np.dot(P_train[:,0:N], vecs.T[0:N,:])) + mu
    new_train = new_train.reshape(image.shape[1],image.shape[2])
    return new_train


def train_dense(train_images, train_labels):
    train_images = train_images/255.0
    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    
    #model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128,activation='relu'))
    #model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dense(2))

    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse'])

    model.fit(train_images[:6000], train_labels[:6000], epochs=30, validation_split=0.2, shuffle=True)

    model.evaluate(train_images[6000:6020], train_labels[6000:6020])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    x, y = pred[0]

    print(x, y, train_labels[-1])
    plt.figure()
    plt.imshow(train_images[-1])
    plt.plot(x, y, "ro")
    plt.plot(train_labels[-1][0], train_labels[-1][1], "go")
    plt.show()

    model.save("dense_canny.keras")

def train_cnn(train_images, train_labels, model_name, split):
    train_images = train_images/255.0
    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    model = keras.Sequential([Input(shape=input_shape)])
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2))
    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

    model.fit(train_images[:split], train_labels[:split], epochs=15, validation_split=0.2, shuffle=True )
    model.evaluate(train_images[split:], train_labels[split:])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    x, y = pred[0]

    print(x, y)
    plt.figure()
    plt.imshow(train_images[-1])
    plt.plot(x, y, "ro")
    plt.plot(train_labels[-1][0], train_labels[-1][1], "go")
    plt.show()

    model.save(model_name)

def run_dense(file_name, canny=False): 
    model = keras.models.load_model("dense_canny.keras")
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    centres = []

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if canny is True:
            edges = cv2.Canny(gray_frame, 50, 200)

    
            edges = np.expand_dims(edges, axis=0)
        
            pred = model.predict(edges, batch_size=1, verbose=0)
        else:
            gray_frame = np.expand_dims(gray_frame, axis=0)
            pred = model.predict(gray_frame, batch_size=1, verbose=0)

        x, y = pred[0]
        centres.append((x, y))

    return np.array(centres)


def run_cnn(file_name, model_name, canny=False):
    model = keras.models.load_model(model_name)
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    centres = []

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame/255.0
        
        if canny is True:
            edges = cv2.Canny(gray_frame, 50, 200)
    
            edges = np.expand_dims(edges, axis=0)
        
            pred = model.predict(edges, batch_size=1, verbose=0)
        else:
            gray_frame = np.expand_dims(gray_frame, axis=0)
            pred = model.predict(gray_frame, batch_size=1, verbose=0)


        x, y = pred[0]
        centres.append((x, y))

    return np.array(centres)

def train_z_cnn(train_images, train_labels, model_name, split):
    train_images = train_images/255.0
    input_shape = (train_images[0].shape[0], train_images[0].shape[1], 1)
    model = keras.Sequential([Input(shape=input_shape)])
    model.add(keras.layers.Conv2D(32, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, 3, activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

    model.fit(train_images[:split], train_labels[:split], epochs=15, validation_split=0.2, shuffle=True )
    model.evaluate(train_images[split:], train_labels[split:])

    test = train_images[-1]
    test = np.expand_dims(test, axis=0)
    pred = model.predict(test, batch_size=1)

    z = pred[0]

    print(z, train_labels[-1][0])

    model.save(model_name)

def cnn_z(file_name, model_name):
    model = keras.models.load_model(model_name)
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    z_vals = []

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame/255.0

        gray_frame = np.expand_dims(gray_frame, axis=0)
        pred = model.predict(gray_frame, batch_size=1, verbose=0)


        z = pred[0]
        z_vals.append(z)

    return np.array(z_vals)
