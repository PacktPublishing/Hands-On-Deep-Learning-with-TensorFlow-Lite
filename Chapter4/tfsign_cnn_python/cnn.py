# -*- coding: utf-8 -*-
"""
Created on Fri June 27 18:10:25 2019

@author: vbhaumik
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.core import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import keras
#Visulization

import glob
import os
root_dir = 'dataset\\training_set\\'
all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
def get_class(img_path):
    return int(img_path.split('\\')[-2])
#all_img_paths = random.shuffle(all_img_paths)
num_images = 16
plt.figure(figsize=(8, 8))
for i in range(num_images):
  plt.subplot(4, 4, i+1)
  img = cv2.imread(all_img_paths[i*80])
  plt.imshow(img, cmap=plt.cm.gray)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("True label: {}".format(get_class(all_img_paths[i*80])))

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(128, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 62, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
#classifier.load_weights("model2.h5")
history= classifier.fit_generator(training_set,
                         steps_per_epoch = 4575,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2520)
#Plot Accuracy vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['acc'], color='b',label='accuracy')
plt.plot(history.history['val_acc'], color='g',label='validation_accuracy')
plt.legend(loc='bottom right', prop={'size': 24})
plt.show()
#Plot Loss vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], color='b',label='Loss')
plt.plot(history.history['val_loss'], color='g',label='Validation Loss')
plt.legend(loc='Top right', prop={'size': 24})
plt.show()

import glob
import os
root_dir = 'dataset\\test_set\\'
all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
def get_class(img_path):
    return int(img_path.split('\\')[-2])
#all_img_paths = random.shuffle(all_img_paths)
import numpy as np
num_images = 16
plt.figure(figsize=(8, 10))
for i in range(num_images):
  plt.subplot(4, 4, i+1)
  img = cv2.imread(all_img_paths[i*100])
  img1 = cv2.resize(img,(64,64))
  img1 = np.expand_dims(img1,axis=0)
  result = classifier.predict_classes(img1)
  plt.imshow(img, cmap=plt.cm.gray)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("True label: {}\n Predicted lable:{}".format(get_class(all_img_paths[i*100]),result[0]))

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/stop.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
import time
start_time=time.time()
result = classifier.predict(test_image)
print("------%s second------"%(time.time()-start_time))


classifier.save("tf_sign_model.h5")
print("Saved model to disk")
keras_model = 'tf_sign_model.h5'
tflite_mnist_model = "tfsign_tflite_model.tflite"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model)
tflite_model = converter.convert()
open(tflite_mnist_model, "wb").write(tflite_model)

import glob
import os
root_dir = 'dataset\\test_set\\'
all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
def get_class(img_path):
    return int(img_path.split('\\')[-2])
y_test = []
y_pred = []
for img_path in all_img_paths:
        test_image = image.load_img(img_path, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        label = get_class(img_path)
        y_test.append(label)
        y = int(classifier.predict_classes(test_image))
        y_pred.append(y)

from sklearn.metrics import confusion_matrix
from sklearn import datasets, metrics 
y_test = np.array(y_test)
y_pred = np.array(y_pred)      
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))
print("classification report for classifier :\n%s\n" % (metrics.classification_report(y_test, y_pred)))
c = confusion_matrix(y_test,y_pred)
print("confusion matrix: ")
print(c)

        