import cv2
import mediapipe as mp
import numpy as np
import requests
from scipy.spatial import distance as dist

# 初始化 Mediapipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 設定安全範圍座標
safe_zone_top_left = (50, 50)
safe_zone_bottom_right = (550, 450)

# LINE Notify Token
line_token = 'JEuuSIfjcfMZ0xsV2fkRSsb5YZP6YIHVRleaNE6YBC0'

# 發送LINE通知
def send_line_message(message):
    headers = {
        "Authorization": "Bearer " + line_token,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {'message': message}
    response = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
    return response

# 計算 EAR 的函數
def calculate_EAR(eye_landmarks):
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 打開攝影機
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose, mp_face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image)
        results_face = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results_pose.pose_landmarks:
            # 繪製姿勢標記
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results_pose.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            
            # 獲取身體中心點
            body_center_x = int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.shape[1])
            body_center_y = int(results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.shape[0])
            
            # 檢查是否在安全範圍內
            if not (safe_zone_top_left[0] < body_center_x < safe_zone_bottom_right[0] and
                    safe_zone_top_left[1] < body_center_y < safe_zone_bottom_right[1]):
                status = "OUTSIDE"
                cv2.putText(image, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                send_line_message("Warning: Baby is out of the safe zone!")
            else:
                status = "INSIDE"
                cv2.putText(image, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                # 左眼和右眼的索引（基於 Mediapipe Face Mesh 標記）
                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]

                left_eye_points = [(int(face_landmarks.landmark[idx].x * image.shape[1]), 
                                    int(face_landmarks.landmark[idx].y * image.shape[0])) for idx in left_eye_idx]
                right_eye_points = [(int(face_landmarks.landmark[idx].x * image.shape[1]), 
                                     int(face_landmarks.landmark[idx].y * image.shape[0])) for idx in right_eye_idx]
                
                # 計算 EAR
                left_ear = calculate_EAR(left_eye_points)
                right_ear = calculate_EAR(right_eye_points)
                ear = (left_ear + right_ear) / 2.0
                
                # 在畫面上顯示 EAR
                cv2.putText(image, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 活動偵測
        fgmask = fgbg.apply(image)
        activity_level = cv2.countNonZero(fgmask)
        if activity_level > 5000:  # 活動量閾值
            cv2.putText(image, "Baby is Active", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 畫出安全範圍
        cv2.rectangle(image, safe_zone_top_left, safe_zone_bottom_right, (0, 255, 0), 2)
        
        cv2.imshow('Baby Monitor', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
