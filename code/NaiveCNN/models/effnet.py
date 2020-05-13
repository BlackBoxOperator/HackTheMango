# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 21:56:37 2020

@author: Vishal

@modifier: nobodyzxc
"""

# Part 1 - Building the CNN
import sys
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_name = 'effnet'
train_path = os.path.join('data', 'C1-P1_Train')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_path,  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 224, 244
        batch_size=batch_size,
        # Specify the classes explicitly
        classes = ['A','B','C'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

import tensorflow as tf
import efficientnet.keras as efn

model = efn.EfficientNetB0(weights=None, classes=3)  # or weights='noisy-student' || 'imagenet'

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

total_sample=train_generator.n

n_epochs = 30

history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(total_sample/batch_size),
        epochs=n_epochs,
        verbose=1)

model.save('{}.h5'.format(model_name))

#import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('mongoA.jpg', target_size = (200,200))
##test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis=0)
#result = model.predict(test_image)

#print(['A', 'B', 'C'][np.where(result[0] == 1)[0][0]])
