import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def LBP_hists(image, n_bins=128):
    '''
    Разбивает изображение на 9 квадратов, каждый из которых (плюс само изображение)
    приводится к 2м цветовым пространствам: YCrCb и HSV и разбивается по каналам.
    Для полученных изображений вычисляются гистограммы local binary patterns.
    Возвращает список гистаграмм и столбцы.

    image - входное изображение, shape=(224, 224, 3), каналы: BGR
    n_bins: int - количество столбцов гистограммы
    '''
    img_crops = [
        cv2.resize(image, (75,75), interpolation=cv2.INTER_LANCZOS4),
        image[:75,:75,:],
        image[:75, 75:150,:],
        image[:75, 150:,:],
        image[75:150, :75,:],
        image[75:150, 75:150,:],
        image[75:150, 150:,:],
        image[150:, :75,:],
        image[150:, 75:150,:],
        image[150:, 150:,:]
    ]

    temp_ycc = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('int16') for img in img_crops]
    temp_hsv = [cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB).astype('int16') for img in img_crops]

    hist_list = []

    for img in temp_ycc:
        for im in cv2.split(img):
            lbp = local_binary_pattern(im, 8, 1)
            hist, bins = np.histogram(lbp, bins=n_bins)
            hist_list.append(hist)

    for img in temp_hsv:
        for im in cv2.split(img):
            lbp = local_binary_pattern(im, 8, 1)
            hist, bins = np.histogram(lbp, bins=n_bins)
            hist_list.append(hist) 

    return hist_list, bins
    
# load data

import joblib

with open('/train_data/train_crop', 'rb') as f:
    train_data = joblib.load(f)
with open('/train_data/labels_train', 'rb') as f:
    labels_train = joblib.load(f)

with open('/train_data/test_crop', 'rb') as f:
    test_data = joblib.load(f)
with open('/train_data/labels_test', 'rb') as f:
    labels_test = joblib.load(f)
    
# Вычисление local binary patterns для тренировочной и тестовой выборок.

            # train_data, test_data - списки трёхканальных изображений с кропами face detection.
            # labels_train, labels_test - метки изображений: 0 - LIVE, 1 - ATTACK    
    
x_train = []
x_test = []

for x in train_data:
    hist, bins = LBP_hists(x, n_bins=128)
    x_train.append(np.hstack([*hist]))

for x in test_data:
    hist, bins = LBP_hists(x, n_bins=128)
    x_test.append(np.hstack([*hist]))
  
  
# Полученные гистограммы могут использоваться в качестве признаков для обучения классификатора 

rf = RandomForestClassifier(100, n_jobs=2)
rf.fit(x_train, y_train)
# rf.score(x_test, y_test) == 0.86 


# Для снижение размера вектора признаков применяется метод главных компонент.

pca = PCA(500)
pca.fit(x_train)

pca_train = pca.transform(x_train)
pca_test = pca.transform(x_test)
