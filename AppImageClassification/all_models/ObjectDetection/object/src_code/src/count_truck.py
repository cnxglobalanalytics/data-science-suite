##this code counts the number of trucks present 
###input:
###output :
## threshold: 80%
## NMS: 0.0
## frames to remeber: 70
## check the size of bounding box , centroid distance : if greater than 200

# import the necessary packages
import numpy as np
import imutils
import time
from scipy import spatial
import cv2
import numpy as  np
import os

#------------------------------------------------------------------------------------------------------------------------
list_of_vehicles = ["truck"]
LABELS = open('C:/vani_daivajna/transurban/src_code/bin/coco.names').read().strip().split('\n') 
weightsPath = 'C:/vani_daivajna/transurban/src_code/bin/yolov4.weights'
configPath = 'C:/vani_daivajna/transurban/src_code/bin/yolo4.cfg'
inputVideoPath="C:/vani_daivajna/transurban/input_video/1min_part2_.mp4"
#inputVideoPath='op4.avi'
OUTPUT_FILE='C:/vani_daivajna/transurban/output_video/threshold/output_1min_part2_thresh86_dist200.avi'

#-------------------------------------------------------------------------------------------------------------------------


probability_minimum=0.80
#print(probability_minimum)
threshold = 0.0
FRAMES_BEFORE_CURRENT = 70
inputWidth, inputHeight = 416, 416

#--------------------------------------------------------------------------------------------------------------------------

def displayVehicleCount(frame, vehicle_count):
    #print("frame is "frame)
   #print("count is",vehicle_count)
    cv2.putText(
        frame, #Image
        
        '(confidence >=80%) Trucks count ' + str(vehicle_count), #Label
        (20, 20), #Position
        cv2.FONT_HERSHEY_SIMPLEX, #Font
        0.8, #Size
        (0, 0xFF, 0), #Color
        2, #Thickness
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        )
    
#-----------------------------------------------------------------------------------------------------------------------------

#Displaying the FPS of the detected video
def displayFPS(start_time, num_frames):
    current_time = int(time.time())
    
    if(current_time > start_time):
        os.system('clear') # Equivalent of CTRL+L on the terminal
        #print("FPS:", num_frames)
        num_frames = 0
        start_time = current_time
    return start_time, num_frames

#-----------------------------------------------------------------------------------------------------------------------
#draw bownding box
def drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame):
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [25,250,250]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            #Draw a green dot in the middle of the box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (x + (w//2), y+ (h//2)), 2, (25, 255, 255), thickness=1)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
#---------------------------------------------------------------------------------------------------------------------------

#Identifying if the current box was present in the previous frames
def boxInPreviousFrames(idxs, boxes, classIDs, confidences, frame,previous_frame_detections, current_box, current_detections):
   # print("lastbox:")
    #print(current_box)
    centerX, centerY, width, height = current_box
    #print("\ncurrent_box")
    #print(current_box)
    dist = np.inf #Initializing the minimum distance
    # Iterating through all the k-dimensional trees
      
    for i in range(FRAMES_BEFORE_CURRENT):
        coordinate_list = list(previous_frame_detections[i].keys())
        #print(coordinate_list)
        if len(coordinate_list) == 0: # When there are no detections in the previous frame
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(coordinate_list).query([(centerX, centerY)])
        #print(temp_dist, index,dist)
        #print("temp_dist "temp_dist)
        #print("/nindex "index)
        drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
        if (temp_dist < dist):
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]
    #print("-----------")
    #print(coord)
    #print(dist)
    #print(width/2)
    #print(height/2)
  
    if (dist > 200):
        ##print(dist)
        #print("false")
        
        
        return False
    
    #print(dist)
    #print("true")
    

    #if (dist > 30):
        #print(dist)
        #return False
    #print("-----------------------")
    #print(dist)
    # Keeping the vehicle ID constant
    current_detections[(centerX, centerY)] = previous_frame_detections[frame_num][coord]
    #print(current_detections[(centerX, centerY)])
    return True
#--------------------------------------------------------------------------------------------------------------------------------

#video writer 
def initializeVideoWriter(video_width, video_height, videoStream):
    # Getting the fps of the source video
    sourceVideofps = videoStream.get(cv2.CAP_PROP_FPS)
    # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter(OUTPUT_FILE, fourcc, sourceVideofps,(video_width, video_height), True)

#----------------------------------------------------------------------------------------------------------------------------------
def count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame):
    current_detections = {}
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indices we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centerX = x + (w//2)
            centerY = y+ (h//2)

            # When the detection is in the list of vehicles, AND
            # it crosses the line AND
            # the ID of the detection is not present in the vehicles
            if (LABELS[classIDs[i]] in list_of_vehicles):
                current_detections[(centerX, centerY)] = vehicle_count 
                #print("----------------------------")
                #print(current_detections)
                if (not boxInPreviousFrames(idxs, boxes, classIDs, confidences, frame,previous_frame_detections, (centerX, centerY, w, h), current_detections)):
                    print("new_vehicle")
                    vehicle_count += 1
               
                    #drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
                    #vehicle_crossed_line_flag += True
                #else:
                    #print("their in previous box")
                    #vehicle_count += 1
                    #ID assigning
                    #Add the current detection mid-point of box to the list of detected items
                    # Get the ID corresponding to the current detection

                ID = current_detections.get((centerX, centerY))
                #ID=ID+1
                #print(ID)
                # If there are two detections having the same ID due to being too close, 
                # then assign a new ID to current detection.
                if (list(current_detections.values()).count(ID) > 1):
                    print("count_due_to_overlap")
                    #ID = ID+1
                    current_detections[(centerX, centerY)] = vehicle_count
                    vehicle_count += 1 
                    
                    #print(vehicle_count)

#Display the ID at the center of the box
                #drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
                #cv2.putText(frame, str(ID), (centerX, centerY),\
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)
                #drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
                #drawDetectionBoxes(idxs, boxes, frame)
    return vehicle_count, current_detections


#----------------------------------------------------------------------------------------------------------------------


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
videoStream = cv2.VideoCapture(inputVideoPath)
video_width = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))


previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]
#previous_frame_detections = [spatial.KDTree([(0,0)])]*FRAMES_BEFORE_CURRENT # Initializing all trees
num_frames, vehicle_count = 0, 0
writer = initializeVideoWriter(video_width, video_height, videoStream)
start_time = int(time.time())
#cnt=0
# loop over frames from the video file stream
while True:
    #cnt+=1
    #print("frame is ",cnt)
    num_frames+= 1
    #print("current frame is:", num_frames)
    # Initialization for each iteration
    boxes, confidences, classIDs = [], [], [] 
    #vehicle_crossed_line_flag = True 
    #print(vehicle_crossed_line_flag)
    #Calculating fps each second
    start_time, num_frames = displayFPS(start_time, num_frames)
    # read the next frame from the file
    (grabbed, frame) = videoStream.read()

    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break


    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inputWidth, inputHeight),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    for output in layerOutputs:
        for i, detection in enumerate(output):
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]


            if confidence > probability_minimum:
                #if (classID==7):
                if (classID!=7):
                    continue

                box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])


                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences,probability_minimum,threshold)
    #print(previous_frame_detections)
    #drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
    vehicle_count, current_detections = count_vehicles(idxs, boxes, classIDs, vehicle_count, previous_frame_detections, frame)
   # print(vehicle_count)
    displayVehicleCount(frame, vehicle_count)
    #drawDetectionBoxes(idxs, boxes, classIDs, confidences, frame)
    # write the output frame to diskd
    writer.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break	

    # Updating with the current frame detections
    previous_frame_detections.pop(0) #Removing the first frame from the list
    # previous_frame_detections.append(spatial.KDTree(current_detections))
    previous_frame_detections.append(current_detections)

        # release the file pointers
#print(vehicle_count)
print("[INFO] cleaning up...")
writer.release()
videoStream.release()


            