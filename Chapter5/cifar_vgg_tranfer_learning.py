# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:15:14 2019

@author: bhaum
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils


#import Dataset
from keras.datasets import cifar10
(X_Train, y_Train), (X_Test, y_Test) = cifar10.load_data()

#Define Labels
class_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","Truck"]


#Visualize dataset
fig = plt.figure(figsize=(12, 12))
for i in range(0, 12):
    img = X_Train[i]
    fig.add_subplot(3, 4, i+1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("True label: {}".format(class_labels[y_Train[i][0]]))
n_classes = len(np.unique(y_Train))
y_Train = np_utils.to_categorical(y_Train, n_classes)
y_Test = np_utils.to_categorical(y_Test, n_classes)

#Normalize Dataset
X_Train = X_Train.astype('float32')/255.
X_Test = X_Test.astype('float32')/255.

#Define Model for feature Extraction
from keras import backend as K
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam


from keras.callbacks import EarlyStopping
K.clear_session()
input_shape = (32, 32, 3)
vgg_base_model = VGG19(weights='imagenet', include_top=False,input_shape= input_shape)

from keras.layers.core import Dropout
x = vgg_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x= Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x= Dropout(0.5)(x)
out = Dense(10, activation='softmax')(x)

model = Model(inputs=vgg_base_model.input, outputs=out)

for layer in vgg_base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
callbacks = [EarlyStopping(monitor='val_acc', patience=20, verbose=0)]
n_epochs = 50
batch_size = 128
history = model.fit(X_Train, y_Train, epochs=n_epochs, batch_size=batch_size, validation_split=0.25, callbacks=callbacks)


#Plot the results
plt.plot(np.arange(len(history.history['acc'])), history.history['acc'], label='training')
plt.plot(np.arange(len(history.history['val_acc'])), history.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy ')
plt.legend(loc=0)
plt.show()

#convert the model to Tensorflow Lite format
import tensorflow as tf
model.save("cifar_model.h5")
print("Saved model to disk")
keras_model = 'cifar_model.h5'
tflite_cifar_model = "cifar_tflite_model.tflite"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model)
tflite_model = converter.convert()
open(tflite_cifar_model, "wb").write(tflite_model)


#Fine tuning
vgg_base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(vgg_base_model.layers))

# Fine tune from 15th layer onwards
fine_tune_at = 15


for layer in vgg_base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(5e-5),
              metrics=['accuracy'])
model.summary()
n_epochs = 50
batch_size = 128
history = model.fit(X_Train, y_Train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, verbose=1, callbacks=callbacks)

plt.plot(np.arange(len(history.history['acc'])), history.history['acc'], label='training')
plt.plot(np.arange(len(history.history['val_acc'])), history.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy ')
plt.legend(loc=0)
plt.show()


import tensorflow as tf
model.save("cifar_fine_tune_model.h5")
print("Saved model to disk")
keras_model = 'cifar_fine_tune_model.h5'
tflite_cifar_finetune_model = "cifar_fine_tune_tflite_model.tflite"
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_model)
tflite_model = converter.convert()
open(tflite_cifar_finetune_model, "wb").write(tflite_model)


from keras.models import load_model
model = load_model('cifar_fine_tune_model.h5')
fig = plt.figure(figsize=(12, 12))
for i in range(0, 12):
    img = X_Test[i]
    y_pred = np.argmax(model.predict(np.expand_dims(X_Test[i],axis=0)))
    fig.add_subplot(3, 4, i+1)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("True label: {}\nPredicted Label: {}".format(class_labels[np.argmax(y_Test[i])],class_labels[y_pred]))

test_loss, test_acc = model.evaluate(X_Test, y_Test)
print('Test accuracy:', test_acc)