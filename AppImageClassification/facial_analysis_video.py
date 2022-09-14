import imutils
import cv2
#import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip
from deepface import DeepFace
from ImageClassification.settings import MODELS_PATH

filepath = MODELS_PATH + '/facial_analysis_video/'
net = cv2.dnn.readNetFromCaffe(filepath+'deploy.prototxt.txt', filepath+'res10_300x300_ssd_iter_140000.caffemodel')
transcribed_audio_file_name = "transcribed_speech.wav"
image_test = filepath+'Sachin_NP_1.png'



def face_movement_detect(video):
    cap = cv2.VideoCapture(video)
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = imutils.resize(frame, width=1000)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            c = 0
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < 0.6:
                    continue
                else:
                    c += 1
                #d['face_detect'] = '1.Human Face detected...'
                    return c
        except AttributeError as ae:
            c = -1
            return c

            # box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            # (startX, startY, endX, endY) = box.astype("int")
            #
            # text = "{:.2f}%".format(confidence * 100)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            #               (0, 0, 255), 2)
            # cv2.putText(frame, text, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


#         try:
#             cv2.imwrite("C:/Users/RNALAB/Documents/ImageClassification_test/media/frame_from_face{0}.jpg".format(count),
#                     frame)
#             count+=1
#             return True
#         except UnboundLocalError:
#             return False

# def facial_movement_det_main(input_video):
#     count = face_movement_detect(input_video)
#
#     try:
#         if count >= 1:
#             d['face_movement'] = "2.Facial movement detected.."
#     except TypeError as te:
#         d['face_movement'] = "1.Face isn't detected!!"


def speech_transcription(input_video):
    audioclip = AudioFileClip(input_video)
    audioclip.write_audiofile(transcribed_audio_file_name)
    r = sr.Recognizer()

    with sr.AudioFile(transcribed_audio_file_name) as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source)
    try:
        l = r.recognize_google(audio)
        return l
    except:
        l=[]
        return l



def doc_verification(video):
    input_video = cv2.VideoCapture(video)
    c = 0
    while (input_video.isOpened()):

        ret, frame = input_video.read()

        if not ret:
            break
        try:
            verification = DeepFace.verify(img1_path=image_test, img2_path=frame)
            if (verification['verified'] != True):
                continue
            else:
                c += 1
                return c

        except ValueError as ve:
            break


def facial_analysis_main(input_video):

    d={}
    count = face_movement_detect(input_video)

    try:
        if count >= 1:
            d['face_detection'] = "Human face detected!"
            d['face_movement'] = "Facial movement detected !"
        elif count == -1:
            d['face_detection'] = "Human face isn't detected!"
            d['face_movement'] = "Facial movement isn't detected !"
    except TypeError as te:
        d['face_detection'] = "Face isn't detected !!"
        d['face_movement'] = "Facial movement isn't detected !!"

    l = speech_transcription(input_video)
    try:
        l_int = [int(i) for i in l]
        if len(l_int) > 0:

            if l_int == sorted(l_int):
                d['speech_result'] = "The input digits from speech in the video are ordered code.."
            else:
                d['speech_result'] = "The input digits from speech in the video aren't ordered.."
        elif len(l_int) == 0:
            d['speech_result'] = "Numerical digit is not recognized in the video.."
    except ValueError as ve:
        d['speech_result'] = "Numerical digit is not recognized in the video.."

    count = doc_verification(input_video)

    try:
        if count >= 1:
            d['doc_verification'] = "Match Found...Face Verified with the submitted document!!"
    except TypeError as te:
        d['doc_verification'] = "Match wasn't found..Unsuccessful Verification!!!"
    return d

