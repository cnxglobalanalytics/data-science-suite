import json
import os, time, sys
from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import default_storage
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from math import sqrt
from PIL import Image
from ImageClassification.settings import BASE_DIR,MEDIA_ROOT

# def export(request):
#     response = HttpResponse(content_type = 'text/csv')
#     response['Content-Disposition'] = 'attachment; filename="csv_database_write.csv"'
#
#     writer = csv.writer(response)
#     writer.writerow(['Image_id', 'Accuracy'])
#
#     for user in users:
#         writer.writerow(user)

    # return response

df = pd.DataFrame(columns=['Image_source','Class_type','Result'])

def cutOff(mask):

    ls_imsource = []
    ls_classtype = []
    ls_result = []

    power = 0.1
    image = MODELS_PATH + "/new_frame0.jpg"
    threshold_frac = 0.01
    ls = []
    image_np = np.array(plt.imread(image))
    mask_np  = np.array(plt.imread(mask))

    mask_background = mask_np[1, 1]
    img_background = image_np[1,1]

    start_row = image_np.shape[0] - mask_np.shape[0]
    start_col = image_np.shape[1] - mask_np.shape[1]

    for row in range(start_row, image_np.shape[0]):
        for col in range(start_col, image_np.shape[1]):
            # reading pixel from mask
            temp_RGB_vector = image_np[row - start_row, col - start_col]
            # measuring distance
            temp_distance = (np.sum(np.absolute(np.subtract(temp_RGB_vector.astype(np.int16), mask_background.astype(np.int16)))))
            # calculating percentage
            percent = temp_distance / sqrt((255) ** 2 + (255) ** 2 + (255) ** 2)
            #print(percent)
            ls.append(percent)

    #printmd("**The sample image frame from given video:**\n",color="blue")

    input_image = Image.open(mask)
    #display(input_image)
    #print("\n\n")
    #printmd("**Does the image contain Logos/Watermarks/Embedded Text ?- (Yes/No)**",color="blue")

    d={}
    d["Language"] ="Yes"
    if np.mean(ls) > threshold_frac :
        #printmd("**Result ::**", color="blue")
        #printmd("**Result :: Yes**", color="green")
        d['is_result'] = "Yes"
        ls_classtype.append("Logo/Watermark/Embedded Text")
        #printmd("**Okay, does the image contain Triller Logos/Watermarks/Embedded Text ?- (Yes/No)**",color="blue")
        if 'Triller' in mask.split(".")[0]:
            # printmd("**Result :: Yes**", color="green")
            d['is_triller'] = "Yes"
            ls_result.append("Triller Logo")
            d['action'] = "Moderation is Required"
        else:
            d['is_triller'] = "No"
            ls_result.append("Non-Triller Logo")
            d['action'] = "Delete Video(Moderation is not Required)"
            #printmd("**Result :: No**", color="red")


    else:
        d['is_result'] = "No"
        #printmd("**Result :: No**", color="red")

    df['Result'] = ls_result
    df["Class_type"] = ls_classtype
    words = str(mask).split("/")
    df["Image_source"] = (words[-1])
    excel_filename = "Image_Classification" + ".xlsx"
    df.to_excel(BASE_DIR+'/media/' + excel_filename)
    d["file_name"] = '/media/' + excel_filename
    pdf_filename = "Content_Moderation_Report.pdf"
    #df.to_excel(BASE_DIR+'/media/' + excel_filename)

    d["pdf_file_name"] = '/media/' + pdf_filename
    #print(d['file_name'])
    #print(BASE_DIR,">>>>>>>>>>>>>>>")
    return d

def delete_old_files(days=10):
    for f in os.listdir(MEDIA_ROOT):        
        if os.stat(os.path.join(MEDIA_ROOT, f)).st_mtime < time.time() - days * 86400:            
            os.remove(os.path.join(MEDIA_ROOT, f))


def dashboard(request):
    #print("Session key:",request.session.session_key)
    delete_old_files()
    
    if request.POST:
        data = {}
        try:
            input_file = request.FILES['file']
            event = request.POST.get('event')
        except:
            return HttpResponse(json.dumps({"error": "File is required."}),
                                content_type="application/json")

        fs = default_storage
        filename = fs.save(input_file.name, input_file)
        file_url = fs.url(filename)
        data["image_url"]=file_url
        input_file = BASE_DIR.replace('\\', '/') + '/' + file_url
        data['event'] = event
        if event == "IC":
            result = cutOff(input_file)
            data['result'] = result

        elif event == "austpost":
            from AppImageClassification import austpost_clf
            result = austpost_clf.run_example(input_file)
            data['result'] = result

        elif event == "nike_fake_detection":
            from AppImageClassification import nike_fake_detection
            result = nike_fake_detection.nike_fake_detection(input_file)
            data['result'] = result

        elif event == "lie_detection":
            from AppImageClassification import Truth_Lie_Detection
            result = Truth_Lie_Detection.final_result(input_file)
            data['result'] = result

        elif event =="OD":
            from AppImageClassification import ObjectDetection
            result,file_name,pdf_file_name = ObjectDetection.detection(input_file)
            if not result:
                data['is_result'] = "No vehicle present."
            else:
                data['file_name'] = file_name
                data['pdf_file_name'] = pdf_file_name

        elif event =="FD":
            from AppImageClassification import Face_Detection
            result,file_name= Face_Detection.face_detection(input_file)
            if not result:
                data['is_result'] = "No Face Detected."
            else:
                data['file_name'] = file_name


        elif event =="TC":
            from AppImageClassification import Truck_Counts
            result = Truck_Counts.detection(input_file)
            # if not result:
            #     data['is_result'] = "No vehicle present."
            # else:
            #     data['file_name'] = file_name
            #     data['pdf_file_name'] = pdf_file_name


        #data['result'] = result

        elif event == "Frame_creation":

           from AppImageClassification import Video_to_frames
           result = Video_to_frames.frame_creation(input_file)
           if not result:
               data['is_result'] = "No Frames created ."
           else:
               data['result'] = result

        elif event =="hatespeech_offensive_detect":
            from AppImageClassification import Hatespeech_offensivelang_detection
            result,file_name = Hatespeech_offensivelang_detection.detector_from_csvfile(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="profane_language_detect":
            from AppImageClassification import Profane_language_detection
            result,file_name = Profane_language_detection.detector_from_csvfile(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="resume_screen":
            from AppImageClassification import Resume_screening
            result,file_name = Resume_screening.screen_main(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="ts_lstm":
            from AppImageClassification import time_series_lstm
            result,file_name = time_series_lstm.ts_lstm(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name


        elif event =="emp_attrition_pred":
            from AppImageClassification import Employee_Attition_pred
            result,file_name = Employee_Attition_pred.attrition_pred(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="medicare_claim_anom_detect":
            from AppImageClassification import Medicare_Claim_Anomaly_Detection
            result,file_name,pdf_file_name = Medicare_Claim_Anomaly_Detection.anomaly_detect(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name
                data['pdf_file_name'] = pdf_file_name


        elif event =="aarp_iforest":
            from AppImageClassification import AARP_anomaly_detection
            result,file_name = AARP_anomaly_detection.aarp_iforest(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="aarp_onesvm":
            from AppImageClassification import AARP_anomaly_detection
            result,file_name = AARP_anomaly_detection.aarp_onesvm(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="multivariate_time_series":
            from AppImageClassification import multivariate_time_series
            result,file_name = multivariate_time_series.multivariate_ts_forecast(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="churn_pred":
            from AppImageClassification import customer_churn
            result,file_name = customer_churn.churn_pred(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name
                #data['pbi_file'] = pbi_file

        elif event == "profane_lang_detect_realtime":
            from AppImageClassification import Profane_lang_detect_realtime
            result = Profane_lang_detect_realtime.detector_from_string(input_file)

            data['result'] = result

        elif event == "topic_model_unsup":
            from AppImageClassification import Topic_unsup
            result = Topic_unsup.topic_extract_unsup(input_file)

            data['result'] = result

        elif event == "spoof_video_check":
            from AppImageClassification import spoof_video_check
            result = spoof_video_check.main_spoof_detect(input_file)

            data['result'] = result

        elif event == "facial_analysis":
            from AppImageClassification import facial_analysis_video
            result = facial_analysis_video.facial_analysis_main(input_file)

            data['result'] = result

        elif event == "facial_analysis_image":
            from AppImageClassification import human_face_analysis_image
            result = human_face_analysis_image.human_facial_analysis_main(input_file)

            data['result'] = result

        elif event == "Topic_Model_Supervised":
            from AppImageClassification import Topic_Model_Supervised
            result,file_name = Topic_Model_Supervised.detector_from_string(input_file)

            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event == "Summ_text":
            from AppImageClassification import Summ_text
            result = Summ_text.summ(input_file)

            data['result'] = result

        elif event =="twitter_sent":
            from AppImageClassification import Twitter_Sentiment_Analysis
            result,file_name = Twitter_Sentiment_Analysis.prediction_from_csvfile(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="basic_sent_analysis":
            from AppImageClassification import Sentiment_Analysis
            result,file_name = Sentiment_Analysis.prediction_from_csvfile(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="Handwritten_Text_extraction":
            from AppImageClassification import Handwritten_Text_extraction
            result,file_name = Handwritten_Text_extraction.transcription(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="text_summ":
            from AppImageClassification import Text_summarization
            result,file_name = Text_summarization.summarization(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event =="text_preprocess":
            from AppImageClassification import text_preprocess_basic
            result,file_name = text_preprocess_basic.preprocessing_main(input_file)
            if not result:
                data['is_result'] = "No file found."
            else:
                data['file_name'] = file_name

        elif event == "ner_extraction":
            from AppImageClassification import ner
            result = ner.ner_extraction(input_file)
            data['result'] = result

        elif event == "theme_extraction":
            from AppImageClassification import Themes
            result = Themes.theme_extraction(input_file)
            data['result'] = result

        



        #print(data)
        #common keys
        return HttpResponse(json.dumps({"success": data}),
                            content_type="application/json")
    else:
        return render(request, 'index.html')

def nlp_view(request):
    return render(request, 'nlp.html')

def quant_view(request):
    return render(request, 'quant_analytics.html')

def fraud_view(request):
    return render(request,'fraud_analytics.html')

