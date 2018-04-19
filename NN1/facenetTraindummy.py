import math
import numpy as np
import os
#import h5py
#import scipy
#import cv2
#from scipy import ndimage
#import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.framework import ops
#from matplotlib.pyplot import imshow
#from preprocessing import *
import pickle

#matplotlib inline
np.random.seed(1)

def create_placeholders_for_training(n_H0, n_W0, n_C0):
    # n_H0 -- scalar, height of an input image
    # n_W0 -- scalar, width of an input image
    # n_C0 -- scalar, number of channels of the input
    # Returns
    # X,Y,Z -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    X = tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C0))
    Y = tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C0))
    Z = tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C0))
    return X, Y, Z

def init_params(n_C0):
    tf.set_random_seed(1)
    conv1 = tf.get_variable("conv1", [7,7,n_C0,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv2a= tf.get_variable("conv2a", [1,1,64,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv2 = tf.get_variable("conv2", [3,3,64,192], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv3a= tf.get_variable("conv3a", [1,1,192,192], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv3 = tf.get_variable("conv3", [3,3,192,384], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv4a= tf.get_variable("conv4a", [1,1,384,384], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv4 = tf.get_variable("conv4", [3,3,384,256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv5a= tf.get_variable("conv5a", [1,1,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv5 = tf.get_variable("conv5", [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv6a= tf.get_variable("conv6a", [1,1,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    conv6 = tf.get_variable("conv6", [3,3,256,256], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #fc1 = tf.get_variable("fc1", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #fc2   = tf.get_variable("fc2", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #fc7128= tf.get_variable("fc7128", [7,7,,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
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
                  #"fc1": fc1,
                  #"fc2": fc2,
                  #"fc7128": fc7128,
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
    embeddings = tf.nn.l2_normalize(A_FC7,None)

    return embeddings

def triplet_loss_debug(y_pred, alpha = 0.5):
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
   
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)))
    
    pos_dist2 = tf.Print(pos_dist, [pos_dist], "pos_dist ")
   
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)))
    
    neg_dist2 = tf.Print(neg_dist, [neg_dist], "neg_dist ")
    
    basic_loss = tf.add(tf.subtract(pos_dist2, neg_dist2) , alpha)
    basic_loss2 = tf.Print(basic_loss, [basic_loss], "basic loss: ")
    
    loss = tf.reduce_sum(tf.maximum(basic_loss2,0.0))
    
    loss2 = tf.Print(loss, [loss], "loss ")
    
    return loss2

def triplet_loss(y_pred, alpha=0.5):
    print(alpha)
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))

    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

batches = 2889
def loadCache(iter):
    with open('./cache/inputs'+str(iter)+'.pkl','rb') as f:  # Python 3: open(..., 'rb')
        anchor,positive,negative = pickle.load(f)
    return anchor,positive,negative

tf.reset_default_graph()
with tf.variable_scope("FaceNet", reuse=tf.AUTO_REUSE):
    x,y,z = create_placeholders_for_training(220,220,3)
    params = init_params(3)
    preds1 = forward_prop(params,x)
    tf.get_variable_scope().reuse_variables()
    preds2 = forward_prop(params,y)
    tf.get_variable_scope().reuse_variables()
    preds3 = forward_prop(params,z)

loss = triplet_loss_debug([preds1,preds2,preds3],0.5)
optim = tf.train.AdagradOptimizer(0.05,name = 'optim').minimize(loss)

init  = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    epochs = 1
    train_cache_file = './faceNet1.meta'
    if os.path.exists(train_cache_file):
        saver.restore(sess, './faceNet1')
    for epoch in range(epochs):
        avgCost = 0
        iters = 2
        for i in range(iters):
            anchor, positive, negative = loadCache(i)
            curr_cost, _ = sess.run([loss, optim], feed_dict={x: anchor, y: positive, z: negative})
            avgCost += curr_cost
            if (i % 100 == 0):
                print("i:" + str(i) + ", loss = " + str(curr_cost))
            # print(embed1)
            # print(embed2)
        avgCost /= iters
        print("Avg Loss :" + str(avgCost))

    #saver.save(sess, './faceNet1')