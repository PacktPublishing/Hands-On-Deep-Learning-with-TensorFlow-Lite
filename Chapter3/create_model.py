# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 05:05:23 2019

@author: bhaum
"""

#import libraries
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Load MNIST Dataset

mnist = tf.keras.datasets.mnist
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()
class_names = ["zero","one","two","three","four","five","six","seven","eight","nine"]

#Visulizing Training Data
print("Data type:", type(X_train))
print("Training Dataset shape:", (X_train.shape))
print("Testing Dataset shape:", (X_test.shape))

#Visulizing single image
plt.figure()
plt.imshow(X_train[5])
plt.colorbar()
plt.grid(False)
plt.xlabel("label: {}".format(Y_train[5]))
plt.show()

#Visulizing multiple images
num_images = 9
plt.figure(figsize=(6, 6))
for i in range(num_images):
  plt.subplot(3, 3, i+1)
  plt.imshow(X_train[i], cmap=plt.cm.gray)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("True label: {}".format(Y_train[i]))
  


#Normalize the Image Data
X_train = X_train / 255.0
X_test = X_test / 255.0

#MNIST Visualization
plt.subplot(1, 2, 1)
plt.xticks([])
plt.yticks([])
plt.xlabel("Inverse")
plt.imshow(X_train[0], cmap=plt.cm.binary)

plt.subplot(1, 2, 2)
plt.xticks([])
plt.yticks([])
plt.xlabel("Original")
plt.imshow(X_train[0], cmap=plt.cm.gray)

#Define ANN Model
from keras.layers import (Activation, Dense, Dropout, Flatten,
                          Lambda, MaxPooling2D)
from keras.models import Sequential
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10,activation="softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Summary of Model
model.summary()
#Train The ANN model for 20 epochs
history= model.fit(X_train, Y_train, epochs=20)

#Plot Accuracy vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['acc'], color='b',label='accuracy')
#plt.plot(history.history['val_acc'], color='g',label='validation_accuracy')
#plt.legend(loc='bottom right', prop={'size': 24})
plt.show()
#Plot Loss vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], color='b',label='Loss')
#plt.plot(history.history['val_loss'], color='g',label='Validation Loss')
#plt.legend(loc='Top right', prop={'size': 24})
plt.show()
#Test the model on Test dataset
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

#Evaluation on a Single Image
img = X_test[3]
example_img_as_input = (np.expand_dims(img,0))
print("Image data shape:", example_img_as_input.shape)
img_prediction = model.predict(example_img_as_input)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("label: {}".format(np.argmax(img_prediction)))
plt.imshow(img, cmap=plt.cm.binary)


#Multiple Images
num_rows = 3
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2  *num_cols, 3*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, num_cols, i+1)
  img_prediction = model.predict(np.expand_dims(X_test[i],0))
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("True label: {}\n Predicted lable:{}".format(Y_test[i],np.argmax(img_prediction)))
  plt.imshow(X_test[i], cmap=plt.cm.binary)
  
#Save Keras Model

keras_mnist_model = 'keras_mnist_model.h5'
model.save(keras_mnist_model)

## To load : model = keras.models.load_model(keras_mnist_model)

#Convert the model to TensorFlow Lite
tflite_mnist_model = "mnist_tflite_model.tflite"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_mnist_model)
tflite_model = converter.convert()
open(tflite_mnist_model, "wb").write(tflite_model)

#Input and Output Details of Tensorflow Lite Model

interpreter = tf.contrib.lite.Interpreter(model_path=tflite_mnist_model)
interpreter.allocate_tensors()

print("*****Input details************")
print("name:", interpreter.get_input_details()[0]['name'])
print("shape:", interpreter.get_input_details()[0]['shape'])
print("type:", interpreter.get_input_details()[0]['dtype'])

print("*****Output details***********")
print("name:", interpreter.get_output_details()[0]['name'])
print("shape:", interpreter.get_output_details()[0]['shape'])
print("type:", interpreter.get_output_details()[0]['dtype'])

print("*****All input Details***** ")
print(interpreter.get_input_details()[0])
print("*****All output Details*****")
print(interpreter.get_output_details()[0])


#Predict the Number using TensorFlow Lite Model
example_img_for_tflite = X_test[1]
example_img_for_tflite = np.expand_dims(example_img_for_tflite,0).astype(np.float32)
print("Input data shape:", example_img_for_tflite.shape)
print("Input data type:", example_img_for_tflite.dtype)

input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], example_img_for_tflite)

interpreter.invoke()

output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("\n\nPrediction results:", output_data)
print("Predicted value:", np.argmax(output_data))
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel("True Label:{}\n Predicted label: {}".format(Y_test[1],np.argmax(output_data)))
plt.imshow(X_test[1], cmap=plt.cm.binary)

