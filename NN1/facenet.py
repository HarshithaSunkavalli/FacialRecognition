import math
import numpy as np
import h5py
import scipy
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle

import random
np.random.seed(1)

from matplotlib.pyplot import imshow
import cv2
import tkinter
from tkinter import messagebox


def create_placeholders_for_predicting(n_H0, n_W0, n_C0):
    # n_H0 -- scalar, height of an input image
    # n_W0 -- scalar, width of an input image
    # n_C0 -- scalar, number of channels of the input
    # Returns
    # X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    return X


def init_params(n_C0):
    tf.set_random_seed(1)
    conv1 = tf.get_variable("conv1", [7, 7, n_C0, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv2a = tf.get_variable("conv2a", [1, 1, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv2 = tf.get_variable("conv2", [3, 3, 64, 192], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv3a = tf.get_variable("conv3a", [1, 1, 192, 192], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv3 = tf.get_variable("conv3", [3, 3, 192, 384], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv4a = tf.get_variable("conv4a", [1, 1, 384, 384], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv4 = tf.get_variable("conv4", [3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv5a = tf.get_variable("conv5a", [1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv5 = tf.get_variable("conv5", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv6a = tf.get_variable("conv6a", [1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    conv6 = tf.get_variable("conv6", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    # fc1 = tf.get_variable("fc1", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # fc2   = tf.get_variable("fc2", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # fc7128= tf.get_variable("fc7128", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    parameters = {"conv1": conv1,
                  "conv2a": conv2a,
                  "conv2": conv2,
                  "conv3a": conv3a,
                  "conv3": conv3,
                  "conv4a": conv4a,
                  "conv4": conv4,
                  "conv5a": conv5a,
                  "conv5": conv5,
                  "conv6a": conv6a,
                  "conv6": conv6,
                  # "fc1": fc1,
                  # "fc2": fc2,
                  # "fc7128": fc7128,
                  }
    return parameters


def forward_prop(parameters, x):
    conv1 = parameters['conv1']
    conv2a = parameters['conv2a']
    conv2 = parameters['conv2']
    conv3a = parameters['conv3a']
    conv3 = parameters['conv3']
    conv4a = parameters['conv4a']
    conv4 = parameters['conv4']
    conv5a = parameters['conv5a']
    conv5 = parameters['conv5']
    conv6a = parameters['conv6a']
    conv6 = parameters['conv6']

    # Conv 1
    Z1 = tf.nn.conv2d(x, conv1, strides=[1, 2, 2, 1], padding='VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    N1 = tf.nn.lrn(P1)

    # Conv 2A
    Z2a = tf.nn.conv2d(N1, conv2a, strides=[1, 1, 1, 1], padding='SAME')
    A2a = tf.nn.relu(Z2a)

    # Conv 2
    Z2 = tf.nn.conv2d(A2a, conv2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    N2 = tf.nn.lrn(A2)
    P2 = tf.nn.max_pool(N2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Conv 3A
    Z3a = tf.nn.conv2d(P2, conv3a, strides=[1, 1, 1, 1], padding='SAME')
    A3a = tf.nn.relu(Z3a)

    # Conv 3
    Z3 = tf.nn.conv2d(A3a, conv3, strides=[1, 1, 1, 1], padding='SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Conv 4a
    Z4a = tf.nn.conv2d(P3, conv4a, strides=[1, 1, 1, 1], padding='SAME')
    A4a = tf.nn.relu(Z4a)

    # Conv 4
    Z4 = tf.nn.conv2d(A4a, conv4, strides=[1, 1, 1, 1], padding='SAME')
    A4 = tf.nn.relu(Z4)

    # Conv 5a
    Z5a = tf.nn.conv2d(A4, conv5a, strides=[1, 1, 1, 1], padding='SAME')
    A5a = tf.nn.relu(Z5a)

    # Conv 5
    Z5 = tf.nn.conv2d(A5a, conv5, strides=[1, 1, 1, 1], padding='SAME')
    A5 = tf.nn.relu(Z5)

    # Conv 6a
    Z6a = tf.nn.conv2d(A5, conv6a, strides=[1, 1, 1, 1], padding='SAME')
    A6a = tf.nn.relu(Z6a)

    # Conv 6
    Z6 = tf.nn.conv2d(A6a, conv6, strides=[1, 1, 1, 1], padding='SAME')
    A6 = tf.nn.relu(Z6)
    P6 = tf.nn.max_pool(A6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flattening
    P6F = tf.contrib.layers.flatten(P6)

    # FC 1
    with tf.variable_scope("fc1") as scope:
        Z_FC1 = tf.contrib.layers.fully_connected(P6F, 32 * 256, activation_fn=None, reuse=tf.AUTO_REUSE,
                                                  scope=tf.get_variable_scope())
        A_FC1 = tf.nn.relu(Z_FC1)
    # Maxout
    # M_FC1 = tf.contrib.layers.maxout(A_FC1,32*128)

    # FC_2
    with tf.variable_scope("fc2") as scope:
        Z_FC2 = tf.contrib.layers.fully_connected(A_FC1, 32 * 256, activation_fn=None, reuse=tf.AUTO_REUSE,
                                                  scope=tf.get_variable_scope())
        A_FC2 = tf.nn.relu(Z_FC2)

    # Maxout
    # M_FC2 = tf.contrib.layers.maxout(A_FC2,32*128)

    # FC_7128
    with tf.variable_scope("fc3") as scope:
        Z_FC7 = tf.contrib.layers.fully_connected(A_FC2, 128, activation_fn=None, reuse=tf.AUTO_REUSE,
                                                  scope=tf.get_variable_scope())
        A_FC7 = tf.nn.relu(Z_FC7)

    # l2 Normalization
    embeddings = tf.nn.l2_normalize(A_FC7)

    return embeddings


tf.reset_default_graph()
with tf.variable_scope("FaceNet", reuse=tf.AUTO_REUSE):
    x = create_placeholders_for_predicting(220, 220, 3)
    params = init_params(3)
    preds1 = forward_prop(params, x)
init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,"./faceNet1")

def getDistanceBetweenEmbeddings(embedding1, embedding2):
    #dist = np.sum(np.square(np.subtract(embedding1,embedding2)))
    dist = np.linalg.norm(embedding1-embedding2)
    return dist

def getEmbeddingsFromImageFile(sess, inputImageLoc):
    inputImage = cv2.imread(inputImageLoc)
    processedImage = np.expand_dims(cv2.resize(inputImage, (220,220),interpolation = cv2.INTER_AREA),axis = 0)
    emb1 = sess.run(preds1,feed_dict = {x:processedImage})
    return emb1

def getEmbeddingsFromImageArray(sess, inputImageArray):
    emb1 = sess.run(preds1,feed_dict = {x:np.expand_dims(inputImageArray,axis = 0)})
    return emb1

database = {}
database["hari"] = getEmbeddingsFromImageFile(sess,"recognizableFaces/hari.jpg")
database["sorkin"] = getEmbeddingsFromImageFile(sess,"recognizableFaces/sorkin.jpg")
#database["shyam"] = getEmbeddingsFromImageFile(sess,"recognizableFaces/shyam.jpg")
database["aaron"] = getEmbeddingsFromImageFile(sess,"recognizableFaces/aaron.jpg")
database["peirsol"] = getEmbeddingsFromImageFile(sess,"recognizableFaces/peirsol.jpg")


def recognizeFromFile(sess, inputImage):
    embedding = getEmbeddingsFromImageFile(sess, inputImage)

    min_dist = 100
    print(embedding)
    for (name, db_emb) in database.items():

        dist = getDistanceBetweenEmbeddings(embedding, db_emb)
        print(name, dist, db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 1:
        print("Not in the database.")
    else:
        output_img = cv2.imread("recognizableFaces/" + identity + ".jpg")
        # plt.subplot(2,1,2)
        imshow(output_img)
        print("It's " + str(identity) + "!!, The distance is " + str(min_dist))
    return min_dist, identity


def recognize(sess, inputImage):
    embedding = getEmbeddingsFromImageArray(sess, inputImage)

    min_dist = 100
    distList = []
    # print(embedding)
    for (name, db_emb) in database.items():

        dist = getDistanceBetweenEmbeddings(embedding, db_emb)
        # print(name,dist,db_emb)
        distList.append(dist)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 1:
        print("Not in the database.",str(distList))
    else:
        output_img = cv2.imread("recognizableFaces/" + identity + ".jpg")
        # plt.subplot(2,1,2)
        imshow(output_img)
        print("It's " + str(identity) + "!!, The distance is " + str(min_dist))
    return min_dist, identity


video = cv2.VideoCapture(0)
# Dump first 10 frames to allow webcam to adjust to lighting
for i in range(10):
    check, frame = video.read()

img_no = 0
frame_count = 0
capture_rate = 30
identity = ""

while True:

    check, frame = video.read()  # Capture image from webcam
    cv2.imshow("FaceNet Facial Recognition", frame)  # Show app
    if identity:
        cv2.putText(frame, identity, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    key = cv2.waitKey(1)  # Hit q to quit
    if key == ord('q'):
        break
    frame_count = frame_count + 1  # Update count of total images captured
    if frame_count % capture_rate == 15:  # Capture only nth image, n = capture_rate
        resized_frame = cv2.resize(frame, (220, 220), interpolation=cv2.INTER_AREA)  # Resize image
        min_dist, identity = recognize(sess, resized_frame)  # Send resized images to CCN, get embeddings
        # nameList=["hari", "akshay", "sathwick"]
        # identity = nameList[random.randint(0,len(nameList)-1)]
        # print(identity, frame_count)

print(identity)
messagebox.showinfo("Results", "Detected face of: "+str())
sess.close()
video.release()  # Release Webcam
cv2.destroyAllWindows()