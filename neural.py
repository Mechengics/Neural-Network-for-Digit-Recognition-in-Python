from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Read the training data
df1 = pd.read_csv("train.csv")

# Convert DataFrame to numpy array
trainarr1 = np.array(df1)
m, n = trainarr1.shape

# Extract labels and data separately
labels = trainarr1[:, 0]  # First column contains labels
data = trainarr1[:, 1:]  # Exclude the first column for data

# Normalize the data
data = data / 255.0

# Transpose the data matrix
trainarr2 = data.T

# Define random generator
def rgenerator():
    # Use He initialization for weights(#This is a life saver for this code)
    w1 = np.random.randn(10, trainarr2.shape[0]) * np.sqrt(2.0 / trainarr2.shape[0])
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)
    b2 = np.zeros((10, 1))
    return w1, b1, w2, b2

# Define the Relu activation function.
def RELU(d1):
    return np.maximum(d1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max(x) for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)

def fprop(w1, b1, w2, b2):
    d1 = w1.dot(trainarr2) + b1  # First dot product
    rd1 = RELU(d1)
    d2 = w2.dot(rd1) + b2
    sd2 = softmax(d2)
    return d1, rd1, d2, sd2

def onehot(labels):
    onehot = np.zeros((labels.size, labels.max() + 1))
    onehot[np.arange(labels.size), labels] = 1
    onehot = onehot.T
    return onehot

# Derivative of Relu activation function
def deriv_relu(a):
    return a > 0

def bprop(d1, w2, rd1, sd2, labels, trainarr2):
    X = labels.size
    onehot1 = onehot(labels)
    error = sd2 - onehot1
    dw2 = 1/X * error.dot(rd1.T)
    db2 = 1/X * np.sum(error, axis=1, keepdims=True)
    dz = w2.T.dot(error) * deriv_relu(d1)
    dw1 = 1/X * dz.dot(trainarr2.T)
    db1 = 1/X * np.sum(dz, axis=1, keepdims=True)
    return dw2, db2, dw1, db1

def update(w1, w2, b1, b2, dw1, db1, dw2, db2, lr):
    w1 = w1 - lr * dw1
    b1 = b1 - lr * db1
    w2 = w2 - lr * dw2
    b2 = b2 - lr * db2
    return w1, b1, w2, b2

def getindex(sd2):
    return np.argmax(sd2, 0)

def accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size
def gradientdescent(trainarr2, labels, iterations, lr):
    w1, b1, w2, b2 = rgenerator()
    for i in range(iterations):
        d1, rd1, d2, sd2 = fprop(w1, b1, w2, b2)
        dw2, db2, dw1, db1 = bprop(d1, w2, rd1, sd2, labels, trainarr2)
        w1, b1, w2, b2 = update(w1, w2, b1, b2, dw1, db1, dw2, db2, lr)
        if i % 10 == 0:
            predictions = getindex(sd2)
            print(f"Iteration {i}: Predictions - {predictions[:10]}")  # Print the first 10 predictions for brevity
            print("ACCURACY IS:", accuracy(predictions, labels))
    return w1, b1, w2, b2

iteration = 1000
learning_rate = 0.01  # Adjust the learning rate if necessary
w1, b1, w2, b2 = gradientdescent(trainarr2, labels, iteration, learning_rate)
