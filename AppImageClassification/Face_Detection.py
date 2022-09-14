import cv2
import pandas as pd
import numpy
from ImageClassification.settings import BASE_DIR


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def read_image(image_path):
    input_image = cv2.imread(image_path)
    return (input_image)

def face_detection(input_path):

    df = pd.DataFrame()
    ls_result = []
    ls_input = []
    img = read_image(input_path)
    print(input_path)
    ls_input.append(str(input_path).split("/")[-1])
    #words = str(input_path).split("/")
    df["Image_source"] = ls_input#(words[-1])
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    if type(faces) == tuple:
        print('no object detected')
        ls_result.append('No Face Detected')
        return False, False
    elif type(faces) == numpy.ndarray:
        ls_result.append('Face Detected')
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(BASE_DIR + '/media/model_output11.jpg', img)




        df['Result'] = ls_result
        excel_filename = "Face_Detection" + ".xlsx"
        df.to_excel(BASE_DIR + '/media/' + excel_filename)
        df_filepath = '/media/' + excel_filename

        return True, df_filepath


