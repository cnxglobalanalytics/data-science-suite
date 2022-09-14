# Importing necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np

labels = open('../Object/bin/coco.names').read().strip().split('\n')  
#loading the yolo3 weihts which has been trained on the MScoco data set
weights_path = '../Object/bin/yolov3.weights'
#loading the configaration file of Yolo3
configuration_path = '../Object/bin/yolov3.cfg'


#setting the minimum probabilty for the prediction inorder to avoid the weak predictions
probability_minimum = 0.5
#setting the threshold for the non maximum suppression 
# this will be helpful in deciding the final bounding box decision for the object that has been detected 
threshold = 0.3


#passing the RGB image 
#reading in the BGR format
def read_image(image_path):
    input_image=cv2.imread(image_path)
    return(input_image)

def shape_of_image(input_image):
    image_input_shape = input_image.shape
    return(image_input_shape)

def preprocessing(image_input):
    #mean_substraction, normalization and RB channels swapping
    #returns : 1.number of image 2.number of color channel 3. width 4 height of the image 
    blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    return(blob)

    
# Passing the weight and configaration of the YOLO3 to the Opencv framework , Darknet Architecture
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
#layers of the our network , which is mentioned in the above cell
layers_names_all = network.getLayerNames() 
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]  

def pass_blob_to_network(blob):
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
    # Showing spent time for forward pass
    print('YOLO v3 took {:.5f} seconds'.format(end - start))
    return(output_from_network)

def detection(input_image_path):
    #input_image_path = filedialog.askopenfilename(initialdir="C:/", title="Select an image", filetypes=(("jpg files", "*.*"),))
    image_input = read_image(input_image_path)
    try:
        image_input_shape =shape_of_image(image_input)
        blob=preprocessing(image_input)
        output_from_network=pass_blob_to_network(blob)
        # Seed the generator - every time we run the code it will generate by the same rules
        # In this way we can keep specific colour the same for every class
        np.random.seed(42)
        colours = np.random.randint(10, 255, size=(len(labels), 3), dtype='uint8')


        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Getting spacial dimension of input image
        h, w = image_input_shape[:2]  # Slicing from tuple only first two elements

        # Check point
        print(h, w)  

        for result in output_from_network:
        # Going through all detections from current output layer
            for detection in result:
                # Getting class for current object
                scores = detection[5:]
                class_current = np.argmax(scores)

                # Getting confidence (probability) for current object
                confidence_current = scores[class_current]

                # Eliminating weak predictions by minimum probability
                if confidence_current > probability_minimum:
                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps center of detected box and its width and height
                    # That is why we can just elementwise multiply them to the width and height of the image
                    box_current = detection[0:4] * np.array([w, h, w, h])

                   #5 elements
                    x_center, y_center, box_width, box_height = box_current.astype('int')
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
                    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
        if len(results) > 0:
            # Going through indexes of results
            for i in results.flatten():
                # Getting current bounding box coordinates
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                # Preparing colour for current bounding box
                colour_box_current = [int(j) for j in colours[class_numbers[i]]]

                # Drawing bounding box on the original image
                cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.2f}'.format(labels[int(class_numbers[i])], confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_input, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        1, colour_box_current, 5)
        cv2.imwrite('../Object/model_output/model_output.jpg', image_input)
    except UnboundLocalError:
        print('no object detected')
        
        
#sample input
input="../Object/sample_images/2.jfif"
detection(input)