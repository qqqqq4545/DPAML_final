import os
import argparse
import cv2
import numpy as np
from keras import models
import sys
import time
from threading import Thread
import importlib.util
from enum import Enum
import math
from PIL import Image, ImageFont, ImageDraw
from statistics import mean
from datetime import datetime
from utils import one_hot_decode, translate_Y, lineNotifyMessage
from PIL import Image, ImageDraw

class BodyPart(Enum):
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,

    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16,
class Position:
    def __init__(self):
        self.x = 0
        self.y = 0
class KeyPoint:
    def __init__(self):
        self.bodyPart = BodyPart.NOSE
        self.position = Position()
        self.score = 0.0

class Person:
    def __init__(self):
        self.keyPoints = []
        self.score = 0.0

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
def sigmoid(x):
        return 1. / (1. + math.exp(-x))

AL_FRAME=20
COUNTER=0
min_threshold = float(0.1)
GRAPH_NAME = 'posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
min_conf_threshold = float(0.4)
resW=640
resH=480
imW, imH = int(resW), int(resH)

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
from PIL import ImageDraw

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,GRAPH_NAME)

# If using Edge TPU, use special load_delegate argument

interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get PoseNet details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
trig=0

# Get DrawModel details
name='DrawModel_64x64.tflite'
interpreter_draw = Interpreter(model_path=name)
interpreter_draw.allocate_tensors()
input_details_draw = interpreter_draw.get_input_details()
output_details_draw = interpreter_draw.get_output_details()

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)
fr=0

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):     

while True:
    fr+=1
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    
    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if input_details[0]['dtype'] == type(np.float32(1.0)):
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    heat_maps = interpreter.get_tensor(output_details[0]['index'])
    offset_maps = interpreter.get_tensor(output_details[1]['index'])
    h_pose=len(heat_maps[0])
    w_pose=len(heat_maps[0][0])
    num_key_points = len(heat_maps[0][0][0])
    # Loop over all detections and draw detection box if confidence is above minimum threshold      
    key_point_positions = [[0] * 2 for i in range(num_key_points)]
    for key_point in range(num_key_points):
        max_val = heat_maps[0][0][0][key_point]
        max_row = 0
        max_col = 0
        for row in range(h_pose):
            for col in range(w_pose):
                heat_maps[0][row][col][key_point] = sigmoid(heat_maps[0][row][col][key_point])
                if heat_maps[0][row][col][key_point] > max_val:
                    max_val = heat_maps[0][row][col][key_point]
                    max_row = row
                    max_col = col
        key_point_positions[key_point] = [max_row, max_col]
        #print(key_point_positions[key_point])
    x_coords = [0] * num_key_points
    y_coords = [0] * num_key_points
    x_coords = [0] * num_key_points
    y_coords = [0] * num_key_points
    confidenceScores = [0] * num_key_points
    positions_list = [] #創建空list
    for i, position in enumerate(key_point_positions):
            position_y = int(key_point_positions[i][0])
            position_x = int(key_point_positions[i][1])
            
            y_coords[i] = (position[0] / float(h_pose - 1) * imH +
                           offset_maps[0][position_y][position_x][i])
            x_coords[i] = (position[1] / float(w_pose - 1) * imW +
                           offset_maps[0][position_y][position_x][i + num_key_points])
            #print('(x,y):',x_coords[i],y_coords[i])
            confidenceScores[i] = heat_maps[0][position_y][position_x][i]
            #print("confidenceScores[", i, "] = ", confidenceScores[i])
            x=int(x_coords[i])
            y=int(y_coords[i])
            #print('x=',x,'y=',y,'confidence=%.2f' %confidenceScores[i])
            positions_list.append((int(x/10),int(y/7.5))) #加入節點進list            
            if(confidenceScores[i]>0.4): 
                cv2.circle(frame, (x,y), 5, (0, 255, 0), cv2.FILLED)
               
    #------------------------------------------------------------------------------------------
    #利用節點資訊畫出火柴人圖
    #------------------------------------------------------------------------------------------
    draw_pic = Image.new('RGB', (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(draw_pic)
    nose = positions_list[0]
    left_eye = positions_list[1]
    right_eye = positions_list[2]
    left_ear = positions_list[3]
    right_ear = positions_list[4]
    left_shoulder = positions_list[5]
    right_shoulder = positions_list[6]
    left_elbow = positions_list[7]
    right_elbow = positions_list[8]
    left_wrist = positions_list[9]
    right_wrist = positions_list[10]
    left_hip = positions_list[11]
    right_hip = positions_list[12]
    left_knee = positions_list[13]
    right_knee = positions_list[14]
    left_ankle = positions_list[15]
    right_ankle = positions_list[16]
    #畫出火柴人
    draw.line([left_ear,left_eye,nose,right_eye,right_ear],fill=(0,0,255),width=4)
    draw.line([nose,left_shoulder,right_shoulder],fill=(0,0,255),width=4)
    draw.line([right_wrist,right_elbow,right_shoulder,right_hip,right_knee,right_ankle],fill=(0,0,255),width=4)
    draw.line([left_wrist,left_elbow,left_shoulder,left_hip,left_knee,left_ankle],fill=(0,0,255),width=4)
    draw.line([right_hip,left_hip],fill=(0,0,255),width=4)
    draw_pic = np.array(draw_pic) #將最後畫好的火柴人圖放入變數draw_pic

    gray = cv2.cvtColor(draw_pic, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    pd_img = cv2.bitwise_not(thresh)
    input_data=pd_img[np.newaxis,:,:,np.newaxis]
    
    # tflite
    input_data = input_data.astype('float32')
    interpreter_draw.set_tensor(input_details_draw[0]['index'], input_data)
    interpreter_draw.invoke()
    prediction = interpreter_draw.get_tensor(output_details_draw[0]['index'])
    
    result = one_hot_decode(prediction)
    detected = translate_Y(result[0])
    
    if detected == 'fall' and trig == 0:
        # lineNotifyMessage()
        trig = 1
    
    pic_resize = cv2.resize(draw_pic, (128, 128), interpolation=cv2.INTER_AREA)
    rows,cols,channels = pic_resize.shape
    frame[0:rows, 0:cols ] = pic_resize     # 將火柴人圖加進去
    
    cv2.putText(frame,detected,(30,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(480,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    cv2.imshow('Object detector', frame)
    #cv2.imshow('Pic', draw_pic)         
    
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1 
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    # Press 'r' to 重啟警示功能
    if cv2.waitKey(1) == ord('r'):
        trig=0
        print('Reset Line warning')
# Clean up
cv2.destroyAllWindows()
videostream.stop()
