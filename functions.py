import numpy as np
import cv2
from lbp import LBP_hists


def square_box(img, x1, y1, x2, y2):
    box_h = y2 - y1
    box_w = x2 - x1
    box_center_x = x1 + box_w // 2
    box_center_y = y1 + box_h // 2
    img_h = img.shape[0]
    img_w = img.shape[1]

    square_side = max(box_h, box_w)

    if square_side > min(img_h, img_w):
        square_side = min(img_h, img_w)
        if img_w < img_h:
            box_center_x = img_w // 2
        else:
            box_center_y = img_h // 2  

    elif box_center_x + square_side // 2 > img_w:
        box_center_x = img_w - square_side // 2

    elif box_center_y + square_side // 2 > img_h:
        box_center_y = img_h - square_side // 2

    X1 = max(box_center_x - square_side // 2, 0)
    X2 = X1 + square_side
    Y1 = max(box_center_y - square_side // 2, 0)
    Y2 = Y1 + square_side

    return img[Y1:Y2, X1:X2, :]


def double_square_box(img, x1, y1, x2, y2):
    box_h = y2 - y1
    box_w = x2 - x1
    box_center_x = x1 + box_w // 2
    box_center_y = y1 + box_h // 2
    img_h = img.shape[0]
    img_w = img.shape[1]

    square_side = max(box_h, box_w) * 2

    if square_side > min(img_h, img_w):
        square_side = min(img_h, img_w)
        if img_w < img_h:
            box_center_x = img_w // 2
        else:
            box_center_y = img_h // 2  

    elif box_center_x + square_side // 2 > img_w:
        box_center_x = img_w - square_side // 2

    elif box_center_y + square_side // 2 > img_h:
        box_center_y = img_h - square_side // 2

    X1 = max(box_center_x - square_side // 2, 0)
    X2 = X1 + square_side
    Y1 = max(box_center_y - square_side // 2, 0)
    Y2 = Y1 + square_side

    return img[Y1:Y2, X1:X2, :]


def get_lbp_features(face_crops):
    lbp_features = []
    for x in face_crops:
        hist, bins = LBP_hists(x, n_bins=128)
        lbp_features.append(np.hstack([*hist]))
    return np.array(lbp_features)


def img_split(image):
    image = cv2.resize(image, (225,225), interpolation=cv2.INTER_LANCZOS4 )
    image = image.reshape(1,225,225,3)
    img_crops = [
        image[:,:75,:75,:],
        image[:,:75, 75:150,:],
        image[:,:75, 150:,:],
        image[:,75:150, :75,:],
        image[:,75:150, 75:150,:],
        image[:,75:150, 150:,:],
        image[:,150:, :75,:],
        image[:,150:, 75:150,:],
        image[:,150:, 150:,:]
    ]
    return np.concatenate([*img_crops], axis=0)
