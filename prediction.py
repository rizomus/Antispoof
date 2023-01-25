# !pip install mediapipe
import cv2
import numpy as np
import mediapipe as mp
from google.colab.patches import cv2_imshow
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
import os
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from functions import double_square_box, img_split, get_lbp_features


with open('pca_transformer', 'rb') as f:
    pca_transformer = joblib.load(f)
model_lbp = load_model('model_lbp.keras')
model_crop = load_model('model_crop.keras')
model_double = load_model('model_double.keras')
model_split = load_model('model_split.keras')


def face_detection(IMAGE_FILES):

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:

        crops = []
        double_crops = []
        split_crops = []
        file_names = []

        for idx, file in enumerate(IMAGE_FILES):

            image = cv2.imread(file)
            
            if not(image is None):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else: 
                print(f'No image {file}')
                continue

            if not results.detections: 
                print(f'{file}: Face not detected')
                continue
            
            for detection in results.detections:
                if not(image is None):
                    x1 = detection.location_data.relative_bounding_box.xmin
                    x2 = x1 + detection.location_data.relative_bounding_box.width
                    x1 = round(image.shape[1] * x1)
                    x2 = round(image.shape[1] * x2)
                    y1 = detection.location_data.relative_bounding_box.ymin
                    y2 = y1 + detection.location_data.relative_bounding_box.height
                    y1 = round(image.shape[0] * y1)
                    y2 = round(image.shape[0] * y2)
                    x1 = max(x1, 0)
                    x2 = min(x2, image.shape[1])
                    y1 = max(y1, 0)
                    y2 = min(y2, image.shape[0])
                    face_crop = image[y1:y2, x1:x2, :]
                    double_crop = double_square_box(image, x1, y1, x2, y2)
                    split_crop = img_split(face_crop)
                    face_crop = cv2.resize(face_crop, (224,224), interpolation=cv2.INTER_LANCZOS4 )
                    double_crop = cv2.resize(double_crop, (224,224), interpolation=cv2.INTER_LANCZOS4 )
                    crops.append(face_crop)
                    double_crops.append(double_crop)
                    split_crops.append(split_crop)
                    file_names.append(file)
                else: 
                    print(f'     FAIL: {file}')

    return crops, double_crops, split_crops, file_names
  

  
def spoofing_detector(image_files, report=True):
    '''
    image_files - список путей к файлам изображений
    report: bool - выводить ли информацию на экран
    Возвращает метки изображений: 0 - LIVE, 1 - ATTACK

    '''    
    crops, double_crops, split_crops, file_names = face_detection(image_files)
    
    if not(crops):
        print('No detections')
        return
    
    lbp_features = []
    lbp_features = get_lbp_features(crops)
    pca_features = pca_transformer.transform(lbp_features)

    crops = np.array(crops)
    crops = preprocess_input(crops)

    double_crops = np.array(double_crops)
    double_crops = preprocess_input(double_crops)

    split_crops = np.array(split_crops)
    split_crops = preprocess_input(split_crops)

    pred_1 = model_lbp.predict(pca_features).round().reshape(-1)
    pred_2 = model_crop.predict(crops)[:,1].round()
    pred_3 = model_double.predict(double_crops)[:,1].round()
    pred_4 = []
    for s in split_crops:
        if model_split.predict(s, verbose=0).round()[:,1].sum() < 5:
            pred_4.append(0)
        else:
            pred_4.append(1)
    pred_4 = np.array(pred_4).astype('float')

    pred = (pred_1 + pred_2 + pred_3 + pred_4) / 4
    mask50 = pred==0.5
    pred[mask50] = pred_2[mask50]
    pred = pred.round().astype('int')
    image_class = {0:'LIVE', 1:'ATTACK'}

    if report:
        print()
        for f, p, in zip(file_names, pred):
            print(f'{f},   -    {image_class[p]}')
        print()
    
    return pred
