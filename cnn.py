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


