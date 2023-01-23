import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import Dense, Activation, Input, GlobalAveragePooling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from dataset_generator import DS_generator
import joblib


'''
Загрузка обучающей выборки:

spoof_train, live_train, img_test, - numpy arrays, shape=(None, 224, 224, 3) - изображения RGB, значения пикселей [0..255]
spoof_train - изображения spoofing атак
live_train - изображения реальных лиц
img_test - тестовая выборка, содержит оба класса
labels_test - numpy arrays, shape=(None, 1) - метки классификатора (0 - LIVE, 1 - ATTACK)

'''

img_test = preprocess_input(img_test)           # нормализация к [-1,1]

train_gen = DS_generator(spoof_train, live_train,           # Генератор тренировочной выборки
                         batch_size=256,                        # смешивает случайные
                         flip_prob=0.5, 
                         mask_prob=0, 
                         increase_mask_prob=True,
                         mask_prob_limit=0.65)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
inp = base_model.output
x = GlobalAveragePooling2D()(inp)
x = Dense(256, 'relu')(x) 
x = Dense(128, 'relu')(x) 
classificator = Dense(2, 'softmax')(x)
model  = Model(inputs=base_model.input, outputs=classificator)

for layer in model.layers[:-14]:
    layer.trainable = False

for layer in model.layers[-14:]:
    layer.trainable = True

model.compile(optimizer=Adam(0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['sparse_categorical_accuracy'])

history = model.fit(train_gen,
                    epochs=50, 
                    validation_data=[img_test, labels_test],
                    validation_batch_size=200,
                    shuffle=True)
