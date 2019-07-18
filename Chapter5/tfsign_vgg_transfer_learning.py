# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:15:14 2019

@author: bhaum
"""
# Importing the Keras libraries and packages

from keras.layers import Dense
from keras.layers.core import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

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
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

from keras.applications.vgg16 import VGG16

CLASSES = 62
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(64,64,3))

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
x = Dense(512)(x)
x = Dropout(0.3)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
   
# transfer learning
for layer in base_model.layers:
    layer.trainable = False
      
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,                         
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
#classifier.load_weights("model2.h5")
history= model.fit_generator(training_set,
                         steps_per_epoch = 143,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 79)
#Plot Accuracy vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['acc'], color='b',label='accuracy')
plt.plot(history.history['val_acc'], color='g',label='validation_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy ')
plt.legend(loc='bottom right', prop={'size': 24})
plt.show()
#Plot Loss vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], color='b',label='Loss')
plt.plot(history.history['val_loss'], color='g',label='Validation Loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss ')
plt.legend(loc='Top right', prop={'size': 24})
plt.show()


base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))
# Fine tune from this layer onwards
fine_tune_at = 10

for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,                         
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

history= model.fit_generator(training_set,
                         steps_per_epoch = 143,
                         epochs = 50,
                         validation_data = test_set,
                         validation_steps = 79)

#Plot Accuracy vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['acc'], color='b',label='accuracy')
plt.plot(history.history['val_acc'], color='g',label='validation_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy ')
plt.legend(loc='bottom right', prop={'size': 24})
plt.show()
#Plot Loss vs No of Epochs
plt.figure(figsize=(20,12))
plt.plot(history.history['loss'], color='b',label='Loss')
plt.plot(history.history['val_loss'], color='g',label='Validation Loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss ')
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
  result = model.predict_classes(img1)
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
result = model.predict(test_image)
print("------%s second------"%(time.time()-start_time))


model.save("tf_sign_vggnet_model.h5")
print("Saved model to disk")
keras_model = 'tf_sign_vggnet_model.h5'
tflite_mnist_model = "tfsign_vggnet_tflite_model.tflite"
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
        y = int(model.predict_classes(test_image))
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

        