import imutils
import cv2 
#import wave, math, contextlib
from deepface import DeepFace
from ImageClassification.settings import MODELS_PATH

filepath = MODELS_PATH + '/human_face_analysis_image/'
net = cv2.dnn.readNetFromCaffe(filepath+'deploy.prototxt.txt', filepath+'res10_300x300_ssd_iter_140000.caffemodel')

image_test_doc_verify = filepath+'Sachin_NP_1.png'


def facial_detect_image(imagefile):
    image = cv2.imread(imagefile)
    frame = imutils.resize(image, width=1000)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    c = 0
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence < 0.6:
            #c = -1
            continue
        else:
            c += 1
    return c

def images_doc_verification(image_test1,image_test2):
    verification = DeepFace.verify(img1_path = image_test1, img2_path = image_test2)
    return verification['verified']

def image_direction(input_image):
    face_cascade = cv2.CascadeClassifier(filepath+'haarcascade_frontalface_default.xml')

    frame = cv2.imread(input_image)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    try:
        height, width, channels = frame.shape
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centre_x = x + w / 2
        centre_y = y + y / 2

        return centre_x,width

    except:
        return -9999,-9999



def human_facial_analysis_main(input_image):

    d={}
    count = facial_detect_image(input_image)

    try:
        if count >= 1:
            d['face_detection'] = "Human face detected!"

        elif count == 0:
            d['face_detection'] = "Human face isn't detected!"

    except TypeError as te:
        d['face_detection'] = "Face isn't detected !!"

    verification_flag = images_doc_verification(input_image,image_test_doc_verify)

    if verification_flag == True:
        d['doc_verification'] = "Match Found...Face Verified with the submitted document!!"
    else:
        d['doc_verification'] = "Match wasn't found..Unsuccessful Verification!!!"

    centre_x,width = image_direction(input_image)

    if abs(centre_x - width) > 150:

        d['face_direction'] ='The face is directed to the left side'
        d['selfie_detection'] = 'The image is a not a selfie'
    elif (abs(centre_x - width) > 90) & (abs(centre_x - width) < 120):
        d['face_direction'] ='The face is directed to the right side'
        d['selfie_detection'] = 'The image is a selfie'
    elif (centre_x == -9999)&(width == -9999):
        d['face_direction'] = 'facial direction was not detected'

    else:
        d['face_direction'] ='The face is front directed'
        d['selfie_detection'] ='The image is a selfie'

    return d

