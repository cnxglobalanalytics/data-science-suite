#
import keras
import cv2                                # Computer Vison Library OpenCV
import numpy as np                        # numpy arrays structure our datasets (numpy instead of tensorflow)
import os
import mediapipe as mp                    # mediapipe (https://google.github.io/mediapipe/)
from math import *
from ImageClassification.settings import BASE_DIR,MODELS_PATH,MEDIA_ROOT

mp_face_mesh = mp.solutions.face_mesh              # FaceMesh Model has -- 478 -- Keypoints 
mp_drawing_styles = mp.solutions.drawing_styles    # Drawing styles
mp_drawing = mp.solutions.drawing_utils            # Drawing utilites

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]


def mediapipe_detection(image,model):                 
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)   # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                     # Image is no longer writeable
    results = model.process(image)                    # Make prediction
    image.flags.writeable = True                      # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    # COLOR CONVERSION RGB 2 BGR
    return image, results


# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio

# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    if 1> distance:
        return 1
    else:
        return distance

# 03. Extract Position Landmarks (Keypoints Values) 

def extract_landmarks(img, results):
    resized_frame = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    mesh_coords = mesh_coord_detection(img, results)
    
    # extract face 478 keyponits
    face = np.array([[res.x, res.y, res.z] for res in results.multi_face_landmarks[0].landmark]).flatten() if results.multi_face_landmarks[0] else np.zeros(478*3)
    
    # extract right-eye 16 keypoints
    right_eye_coords = np.array([mesh_coords[p] for p in RIGHT_EYE]).flatten() if results.multi_face_landmarks[0] else np.zeros(16*2)  
    
    # extract left-eye 16 keypoints
    left_eye_coords = np.array([mesh_coords[p] for p in LEFT_EYE]).flatten() if results.multi_face_landmarks[0] else np.zeros(16*2)     
     
    # calculating eye ratio
    ratio = np.array(blinkRatio(resized_frame, mesh_coords, RIGHT_EYE, LEFT_EYE)).flatten() if results.multi_face_landmarks[0] else np.zeros(0)   
    
    # return all feature values 
    return np.concatenate([face,right_eye_coords,left_eye_coords,ratio])


def mesh_coord_detection(img, results):
    img_height, img_width= img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark] if results.multi_face_landmarks[0] else np.zeros(478*2)
    return mesh_coord



actions = np.array(['true', 'lie'])       


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(
            output_frame, actions[num].capitalize()+ ' ' +str(int(prob*100))+'%',
            (3, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

model =  keras.models.load_model(MODELS_PATH +'/Truth_Lie_Detection/truth_lie_detection.h5')

#VDO_PATH = 'Datasets/' + 'lie' +'/' + str(0) + '/' + str(0) + '.mp4'
#VDO_PATH

def final_result(VDO_PATH):
    print("Running ---;;")

    d = {}

    sentence = []
    sequence = []
    threshold = 0.5
    count = 0

    final_result_list = []
    cap = cv2.VideoCapture(VDO_PATH)

    saved_video_name = VDO_PATH.split("/")[-1].split(".")[0]

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    video_save = cv2.VideoWriter(MEDIA_ROOT +'/lie_detection_{}.avi'.format(saved_video_name), 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                image, results = mediapipe_detection(frame, face_mesh)
                try:
                    keypoints = extract_landmarks(image , results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                except:
                    pass
                   
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    final_result_list.append(actions[np.argmax(res)])
                    image = prob_viz(res, actions, image, colors)

                #cv2.imwrite('AppImageClassification/results/Lie_Detection/frame_'+str(count)+'.jpg', image)
                #cv2.imshow('Lie Detecion',image)
                video_save.write(image)
                count += 1
                #if cv2.waitKey(10) & 0xff == ord('q'): # press ESC to exit
                    #break
            else:
                break
            # Break gracefully
            #if cv2.waitKey(10) & 0xff == ord('q'): # press ESC to exit
                #break
        
        Truth = (final_result_list.count('true')/len(final_result_list))*100
        Lie = (final_result_list.count('lie')/len(final_result_list))*100
        truth_lie_dict = {"Truth":Truth,"Lie":Lie}

        print("Truth Percentage : {:.2f} %".format(Truth))
        print("Lie Percentage : {:.2f} %".format(Lie))
        print("Final Output : {}".format("Lie" if Lie >= 15 else "Truth"))
        
        cap.release()
        cv2.destroyAllWindows()

        #d["result"] = 
        d["Truth_percentage"] = "{:.2f} %".format(Truth)
        d["Lie_percentage"] = "{:.2f} %".format(Lie)
        d["Final_Result"] = "Lie" if Lie >= 15 else "Truth"

        mystr =  "This Video is {} Truth and {} Lie".format(d["Truth_percentage"], d["Lie_percentage"])
        d["mystr"] = mystr
        print(d)
        return d
#VDO_PATH = '0.mp4'
#final_result(VDO_PATH)
