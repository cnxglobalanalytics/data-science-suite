import pandas as pd
import cv2
from datetime import datetime

webcams_choice=['First_webcam','Second_webcam']
working_webcam = 'First_webcam'

def webcam_info(filename):
    exceldata = pd.read_excel(filename)
    x = exceldata[exceldata['webcam'] == working_webcam]['id'].iloc[0]
    return x


def spoof_detect(working_webcam_id):
    # define a video capture object
    vid = cv2.VideoCapture(working_webcam_id)

    while (True):
        c = 0

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        if ret == True:
            c += 1

        return c

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def main_spoof_detect(filename):
    d = {}
    d['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d['Frames_Per_Second'] = cv2.CAP_PROP_FPS
    #jsondata = webcam_info_json(filename)
    working_webcam_id = webcam_info(filename)
    d['Working_webcam'] =working_webcam
    flag = spoof_detect(working_webcam_id)
    if flag >=1 :
        d["result"] = "The video is live, non-spoof"
    else:
        d["result"] = "The video is a spoof"

    return d


