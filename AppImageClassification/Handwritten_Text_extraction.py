import os
import easyocr
import PIL
from PIL import ImageDraw
import pandas as pd
import cv2
from ImageClassification.settings import BASE_DIR

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
reader = easyocr.Reader(['en'], gpu=False)

n_top_words = 15

def read_image(file):
    im = PIL.Image.open(file)
    #im = cv2.imread(file)
    return im


def read_bounds(file):
    bounds = reader.readtext(file)
    return bounds

def draw_boxes(image, bounds, color='blue', width=3):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3 ,*p0], fill=color, width=width)
    return image

def transcription(file):
    bounds = read_bounds(file)
    im = read_image(file)

    img1 = draw_boxes(im, bounds)
    filename = BASE_DIR + '/media/saved_image.jpg'
    img1.save(filename)
    img = cv2.imread(filename)
    cv2.imwrite(BASE_DIR + '/media/model_output112.jpg', img)

    Text = []
    Score = []

    data = pd.DataFrame(columns=["Retrieved_Text", "Confidence_Score"])
    for i in range(0, len(bounds)):
        Text.append(bounds[i][1])
        Score.append(bounds[i][2])
    data['Retrieved_Text'] = Text
    data['Confidence_Score'] = Score
    df = data.copy()
    #df = data.sort_values(by='Confidence_Score', ascending=False).iloc[:n_top_words]

    excel_filename = "Handwritten_Text_Extraction" + ".xlsx"
    df.to_excel(BASE_DIR + '/media/' + excel_filename)
    df_filepath = '/media/' + excel_filename

    return True, df_filepath
