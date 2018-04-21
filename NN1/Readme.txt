To run the code, 

First Download the LFW zip file from http://vis-www.cs.umass.edu/lfw/lfw.tgz and Create a "Datasets" folder in the NN1 directory and
extract the lfw zip file downloaded here. In the end, you should have all the image folders of the dataset in the "NN1/Datasets/lfw/" directory

Now install anaconda, scipy, tensorflow, cv2, pymongo, pickle and other required packages in python3.

create an empty "cache2" directory in NN1 folder and run preprocessing.py for populating the cache2 directory with pickle files which 
are the processed numpy arrays of triplets. To run this, make sure you are able to access the mongo db instance present in a
NCSU's VCL machine.

Now run facenetTrain.py to train the model with 445 batches of triplets each of size 64. This trains the model for 40 epochs.
These parameters can be changed by modifying the respective variables in the file.

now populate the images of the people you want to recognize in the recognizableFaces directory and set the image names to the 
identity of the person. now populate the database in the code at line 202-207 with required entries. Database key should be set to
identity of the person which also happens to be the name of the file placed in the directory. Fill the file name appropriately.

Now run facenet.py to launch the facenet application and place your face in the box showed in the UI. The name recognized is displayed in 
the another window popped up along with the live video display window.



