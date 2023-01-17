import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Input, GlobalAveragePooling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from dataset_generator import DS_generator


'''
Загрузка обучающей выборки:

train_crop, test_crop - numpy arrays, shape=(None, 224, 224, 3) - изображения BGR, значения пикселей [0..255]
test_crop = preprocess_input(test_crop) - нормализация к (-1,1)
labels_train, labels_test - numpy arrays, shape=(None, 1) - метки классификатора (0 - LIVE, 1 - ATTACK)

'''

base_model = MobileNet(weights='imagenet', include_top=False)

inp = base_model.output
x = GlobalAveragePooling2D()(inp)
x = Dense(256, 'relu')(x) 
x = Dense(128, 'relu')(x) 
classificator = Dense(1, 'sigmoid')(x)

model  = Model(inputs=base_model.input, outputs=classificator)

for layer in model.layers[:-14]:
    layer.trainable = False

for layer in model.layers[-14:]:
    layer.trainable = True

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen,
                    epochs=75, 
                    batch_size=512, 
                    validation_data=[test_crop, labels_test],
                    validation_batch_size=700,
                    shuffle=True)
