import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import threading
import tensorflow as tf
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
# from telegram_notification import send_telegram_Wakeup
# from telegram_notification import send_telegram_Outside
# from telegram_notification import send_telegram_Moving
# from voice import voice_Alert_Wakeup
# from voice import voice_Alert_Outside
# from voice import voice_Alert_Moving

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


class Detect_final():
    def __init__(self):
        self.i = 0
        self.j = 0
        self.k = 0        
        self.threshold = 0.12
        self.frame_check_ouside = 10
        self.frame_check_eye = 30
        self.frame_check_body = 30
        self.flag = 0
        self.sequence = []
        self.ear_list = []
        self.sentence_lable = []
        self.label = "Waiting"
        self.label_eye = "Waiting"
        self.label_outside = "Waiting"        
        self.flag_ouside = 0
        self.model = tf.keras.models.load_model("model.h5")
        self.model_ear = tf.keras.models.load_model("model.h5")
        #Define 12 eye landmark
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.model_eye = get_mediapipe_eye()
        self.model_body = get_mediapipe_body()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)
        self.last_alert = None
        self.alert_telegram_each = 15
        
        #gpt
        self.lock = threading.Lock()
        self.label_outside = "INSIDE"


    # def draw_prediction(self, centroid, points):
    #     if isInside(points, centroid):
    #         pass
    #         #print("INSIDE")
    #     else:
    #         pass 
    #         #print("OUTSIDE")
    #     return isInside(points, centroid)
    
    def isInside(self, points, centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        return polygon.contains(centroid)
    
    def draw_prediction(self, centroid, points):
        if self.isInside(points, centroid):
            print(f"Centroid {centroid} is INSIDE the polygon.")
            return True
        else:
            print(f"Centroid {centroid} is OUTSIDE the polygon.")
            return False

    
    def isInside(self, points, centroid):
        polygon = Polygon(points)
        centroid = Point(centroid)
        inside = polygon.contains(centroid)
        print(f"Polygon: {polygon}, Centroid: {centroid}, Inside: {inside}")
        return inside

    def alert_Wakeup(self):
        thread = threading.Thread(target=send_telegram_Outside())
        thread1 = threading.Thread(target=voice_Alert_Outside())
        thread2 = threading.Thread(target=send_telegram_Moving())
        thread3 = threading.Thread(target=voice_Alert_Moving())
        thread4 = threading.Thread(target=send_telegram_Wakeup())
        thread5 = threading.Thread(target=voice_Alert_Wakeup())
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            if self.label == "MOVING" and self.label_outside == "OUTSIDE" and self.label_eye == "WAKE UP":
                thread.start()
                thread1.start()
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
            elif self.label == "MOVING" and self.label_outside == "OUTSIDE" and self.label_eye == "SLEEPING":
                thread.start()
                thread1.start()
            elif self.label == "MOVING" and self.label_outside == "INSIDE" and self.label_eye == "WAKE UP":
                thread2.start()
                thread3.start()
                thread4.start()
                thread5.start()
            elif self.label == "MOVING" and self.label_outside == "INSIDE" and self.label_eye == "SLEEPING":
                thread2.start()
                thread3.start()  
            elif self.label == "NO MOVING" and self.label_outside == "OUTSIDE" and self.label_eye == "WAKE UP":
                thread.start()
                thread1.start()
                thread4.start()
                thread5.start()
            elif self.label == "NO MOVING" and self.label_outside == "INSIDE" and self.label_eye == "WAKE UP":
                thread4.start()
                thread5.start()
            else: 
                pass            
              
    def alert_Wakeup(self):
        # cv2.putText(frame, "OUTSIDE", (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            thread = threading.Thread(target=send_telegram_Wakeup())
            thread1 = threading.Thread(target=voice_Alert_Wakeup())
            thread.start()
            thread1.start()

    
    def alert_Outsize(self):
        # cv2.putText(frame, "OUTSIDE", (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            thread = threading.Thread(target=send_telegram_Outside())
            thread1 = threading.Thread(target=voice_Alert_Outside())
            thread.start()
            thread1.start()

    
    def alert_Moving(self,frame):
        if (self.last_alert is None) or (
                (datetime.datetime.utcnow() - self.last_alert).total_seconds() > self.alert_telegram_each):
            self.last_alert = datetime.datetime.utcnow()
            thread = threading.Thread(target=send_telegram_Moving())
            thread1 = threading.Thread(target=voice_Alert_Moving())
            thread.start()
            thread1.start()
        return frame

    def draw_class_on_image_outside(self, label, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (250, 70)
        fontScale = 1
        fontColor = (0, 0, 255)
        thickness = 2
        lineType = 2
        cv2.putText(frame, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return frame
        
    def draw_class_on_image(self, label, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 0, 255)
        thickness = 2
        lineType = 2
        cv2.putText(frame, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return frame
    
    def draw_class_on_image_eye(self, label, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 80)
        fontScale = 1
        fontColor = (0, 0, 255)
        thickness = 2
        lineType = 2
        cv2.putText(frame, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return frame
    
    # def detect_outside(self, test_point3, test_point2, test_point1, test_point5, test_point4, points,frame):
    #     if (self.draw_prediction(test_point3, points) == False 
    #         or self.draw_prediction(test_point2, points) == False 
    #         or self.draw_prediction(test_point1, points) == False 
    #         or self.draw_prediction(test_point4, points) == False
    #         or self.draw_prediction(test_point5, points) == False):
            # cv2.imwrite("alert_outside.png", cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)) (原本就註解了)
            # print("save_outside")                                                             (原本就註解了)
            # t4 = threading.Thread(target = self.alert_Wakeup())                               (原本就註解了)
            # t4.start()                                                                        (原本就註解了)
        #     self.label_outside = "OUTSIDE"                   
        # else:
        #     self.label_outside = "INSIDE"
        #     self.flag_ouside = 0                  
        # return self.label_outside

    def detect_outside(self, test_point3, test_point2, test_point1, test_point5, test_point4, points, frame):
        outside = any(not self.draw_prediction(pt, points) for pt in [test_point3, test_point2, test_point1, test_point4, test_point5])
        with self.lock:
            if outside:
                self.label_outside = "OUTSIDE"
            else:
                self.label_outside = "INSIDE"
        print(f"Detection result: {self.label_outside}")



    def detect(self, model, lm_list, frame):
        #global label      
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        results = model.predict(lm_list)
        print(results[0][0])
        if results[0][0] > 0.5:
            self.label = "BODY MOVING"
            # cv2.imwrite("alert_moving.png", cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))
            # print("save_moving")
            # t5 = threading.Thread(target = self.alert_Wakeup())
            # t5.start()
        else:
            self.label = "NO MOVING"        
        return self.label
    
    def detect_eye(self, model, lm_list, frame):
        lm_list = np.array(lm_list) 
        lm_list = np.expand_dims(lm_list, axis=0)
        results_eye = model.predict(lm_list)
        print(results_eye[0][0])
        if results_eye[0][0] > 0.5:
            self.label_eye = "WAKE UP"
            # cv2.imwrite("alert_wakeup.png", cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))
            # print("Save_wakeup")
            # t6 = threading.Thread(target = self.alert_Wakeup())
            # t6.start()
        else:
            self.label_eye = "SLEEPING"
        return self.label_eye
    
    def mediapipe_detection(self, frame, model_eye, model_body):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        frame.flags.writeable = False                  # Image is no longer writeable
        results_eye = self.model_eye.process(frame)                 # Make prediction
        results_body = self.model_body.process(frame)
        frame.flags.writeable = True                   # Image is now writeable 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return frame, results_eye, results_body
            
    def draw_styled_landmarks(self, image, results):
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        return pose
    
    def output(self, frame, points):
        """
        This function is used to implement our Drowsy detection algorithm
        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.
        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """
        # To improve performance,
        # mark the frame as not writeable to pass by reference.

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        frame.flags.writeable = False                  # Image is no longer writeable
        results_eye = self.model_eye.process(frame)    # Make prediction
        results_body = self.model_body.process(frame)
        frame.flags.writeable = True                   # Image is now writeable 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        frame_h, frame_w, _ = frame.shape
        self.draw_styled_landmarks(frame, results_body)
        self.i += 1
        print(self.i)
        # #1.Check wakeup or sleeping
        if results_eye.multi_face_landmarks:
            # start = datetime.now()
            for face_landmarks in results_eye.multi_face_landmarks:
                if face_landmarks:
                    landmarks = face_landmarks.landmark
                    EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h)
                    EAR_display = format(EAR, '.2f')
                    cv2.putText(frame, "Ratio: {}".format(EAR_display), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)            
                    EAR = np.array([EAR],ndmin=1)
                    self.ear_list.append(EAR)
                    frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])
                    frame = cv2.flip(frame, 1)
                    if len(self.ear_list) == self.frame_check_eye:
                        # predict
                        t1 = threading.Thread(target=self.detect_eye, args=(self.model_ear, self.ear_list, frame))
                        t1.start()
                        self.ear_list = []             
        frame = self.draw_class_on_image_eye(self.label_eye, frame)
            
        #2. Check moving or not
        # start1 = datetime.now()    
        keypoints = self.extract_keypoints(results_body)
        self.sequence.append(keypoints)
        
        #Detect action
        if len(self.sequence) == self.frame_check_body:
            # predict
            t2 = threading.Thread(target=self.detect, args=(self.model, self.sequence, frame))
            t2.start()         
            self.sequence = []                       
        frame = self.draw_class_on_image(self.label, frame)

    
    #3. Check OUTSIDE
        if results_body.pose_landmarks and results_body.pose_landmarks.landmark:
            landmarks = results_body.pose_landmarks.landmark
            NOSE = [landmarks[mp_holistic.PoseLandmark.NOSE.value].x, landmarks[mp_holistic.PoseLandmark.NOSE.value].y]
            RIGHT_INDEX = [landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_INDEX.value].y]
            LEFT_INDEX = [landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_INDEX.value].y]
            RIGHT_FOOT_INDEX = [landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            LEFT_FOOT_INDEX = [landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            self.test_point1 = np.multiply(RIGHT_INDEX, [frame_w, frame_h]).astype(int)
            self.test_point2 = np.multiply(LEFT_INDEX, [frame_w, frame_h]).astype(int)
            self.test_point3 = np.multiply(NOSE, [frame_w, frame_h]).astype(int)
            self.test_point4 = np.multiply(RIGHT_FOOT_INDEX, [frame_w, frame_h]).astype(int)
            self.test_point5 = np.multiply(LEFT_FOOT_INDEX, [frame_w, frame_h]).astype(int)
            
            t3 = threading.Thread(target = self.detect_outside , args = (self.test_point3, self.test_point2, self.test_point1, self.test_point5, self.test_point4, points, frame))
            t3.start()
 
        frame = self.draw_class_on_image_outside(self.label_outside, frame)  
            # Flip the frame horizontally for a selfie-view display.
        return frame

            
def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_mediapipe_eye(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return face_mesh
def get_mediapipe_body():
    model_body = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return model_body

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image



#Check inside or outside/ return True or False
def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    #print(polygon.contains(centroid))
    return polygon.contains(centroid)



# def main():
#     video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
#     detector = Detect_final()

#     # Example: Define points. Adjust this according to your requirements.
#     # For example, it could be a bounding box or specific coordinates from a setup.
#     points = [(100, 100), (200, 100), (200, 200), (100, 200)]  # Example coordinates

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break

#         # Process the frame using your detection methods
#         processed_frame = detector.output(frame, points)  # Pass points here

#         # Display the processed frame
#         cv2.imshow('Frame', processed_frame)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture when everything is done
#     video_capture.release()
#     cv2.destroyAllWindows()

def draw_polygon(frame, points):
    """
    Draws a polygon on the frame based on given points.

    Args:
    frame (np.array): The image frame on which to draw.
    points (list of tuples): List of (x, y) tuples defining the polygon vertices.

    Returns:
    np.array: The modified frame with the polygon drawn on it.
    """
    # Ensure the points are in a NumPy array shape that OpenCV expects
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon on the frame
    # The third parameter is a boolean indicating whether the polygon should be closed: True means close the shape
    cv2.polylines(frame, [pts], True, (255, 0, 0), 3)
    return frame

def main():
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
    detector = Detect_final()

    points = [(50, 50), (500, 50), (500, 800), (50, 800)]  # Define the polygon corners

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = draw_polygon(frame, points)  # Draw the polygon on each frame

        # Process the frame using your detection methods
        processed_frame = detector.output(frame, points)

        # Display the processed frame
        cv2.imshow('Frame', processed_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when everything is done
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# 傳送訊息到Line2
# import requests
# def detect_eye(self, model, lm_list, frame):
#     lm_list = np.array(lm_list)
#     lm_list = np.expand_dims(lm_list, axis=0)
#     results_eye = model.predict(lm_list)
#     if results_eye[0][0] > 0.5:
#         self.label_eye = "WAKE UP"
#         # 發送 LINE 通知
#         send_line_message("警告：寶寶起床 !")
#     else:
#         self.label_eye = "SLEEPING"
#     return self.label_eye

# def detect_outside(self, test_point3, test_point2, test_point1, test_point5, test_point4, points, frame):
#     if (not self.isInside(points, test_point3) or not self.isInside(points, test_point2) or
#         not self.isInside(points, test_point1) or not self.isInside(points, test_point4) or
#         not self.isInside(points, test_point5)):
#         self.label_outside = "OUTSIDE"
#         # 發送 LINE 通知
#         send_line_message("警告：寶寶危險！")
#     else:
#         self.label_outside = "INSIDE"
#     return self.label_outside

# def send_line_message(msg):
#     line_token = "2t5fOLbye8FV3SR1ZT6F6vtcBpLTd8MhUaJx2XWhhs8"
#     url = "https://notify-api.line.me/api/notify"
#     headers = {
#         "Authorization": "Bearer " + line_token
#     }
#     data = {
#         "message": msg
#     }
#     response = requests.post(url, headers=headers, data=data)
#     print(response.text)



#傳送訊息到Line1

# trig=0
# def lineNotifyMessage():
#     # 傳送訊息字串
#     message = "Some one fall!!"    
#     # 修改成你的Token字串
#     token = '2t5fOLbye8FV3SR1ZT6F6vtcBpLTd8MhUaJx2XWhhs8'
    
#     headers = { "Authorization": "Bearer " + token,"Content-Type" : "application/x-www-form-urlencoded" }
#     payload = {'message': message}
#     r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
#     return r.status_code    


# def translate_Y(yi , show = False):
    
#     num = {
#         0 : "fall", 1 : "normal", 2 : "unknow" } 
    
#     yi_new = num.get( yi ) 
            
#     if show:
#         print(yi)
#         print(yi_new)            
        
#     return yi_new

# def one_hot_decode(data):
#     data = np.squeeze(data)
#     list_max=data.tolist()
#     list_max.sort(key=lambda x: float(x), reverse = True)
#     n = np.argwhere(data==list_max[0])
#     return int(n), list_max[0]

# result = one_hot_decode(prediction)
# detected = translate_Y(result[0])


# if detected == 'fall' and trig == 0:
#     lineNotifyMessage()
#     trig = 1
# if cv2.waitKey(1) == ord('q'):
#     break
#     # Press 'r' to 重啟警示功能
# if cv2.waitKey(1) == ord('r'):
#     trig=0
#     print('Reset Line warning')

# cv2.destroyAllWindows()
# videostream.stop()

