!pip install mediapipe
import mediapipe as mp

!git clone https://github.com/rizomus/Antispoof.git

!mv Antispoof/* /content/

from prediction import spoofing_detector
import os

# upload IMGS foder with images

IMGS = os.listdir('IMGS')
IMGS = ['IMGS/' + im for im in IMGS]

spoofing_detector(IMGS)
