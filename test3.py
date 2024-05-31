import cv2
import mediapipe as mp
import numpy as np
import requests
from tensorflow.keras.models import load_model
import collections

# 加載訓練好的多輸出LSTM模型
model = load_model('multi_output_lstm_baby_monitor_model.h5')

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils 

# 設定安全範圍座標
safe_zone_top_left = (50, 50)
safe_zone_bottom_right = (550, 450)

# LINE Notify Token
line_token =  'JEuuSIfjcfMZ0xsV2fkRSsb5YZP6YIHVRleaNE6YBC0'

# 發送LINE通知
def send_line_message(message):
    headers = {
        "Authorization": "Bearer " + line_token,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {'message': message}
    response = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
    return response

# EAR計算
def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# 打開攝影機
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

# 初始化時間序列緩衝區
sequence_length = 10
frame_buffer = collections.deque(maxlen=sequence_length)
awake_status_notified = False

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 姿勢偵測
        pose_results = pose.process(image_rgb)
        face_results = face_mesh.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if pose_results.pose_landmarks:
            # 繪製姿勢標記
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
            
            # 獲取身體中心點
            body_center_x = int(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.shape[1])
            body_center_y = int(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.shape[0])
            
            # 創建ROI
            roi = image[safe_zone_top_left[1]:safe_zone_bottom_right[1], safe_zone_top_left[0]:safe_zone_bottom_right[0]]
            roi_resized = cv2.resize(roi, (64, 64))
            roi_resized = roi_resized / 255.0
            roi_flattened = roi_resized.flatten()
            frame_buffer.append(roi_flattened)
            
            if len(frame_buffer) == sequence_length:
                sequence = np.expand_dims(np.array(frame_buffer), axis=0)
                predictions = model.predict(sequence)
                
                action_prediction = np.argmax(predictions[0][0])
                pacifier_prediction = predictions[1][0]
                active_prediction = predictions[2][0]
                
                actions = ["Climbing", "Laying", "Normal", "Obstructed"]
                action_text = actions[action_prediction]
                cv2.putText(image, action_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if pacifier_prediction > 0.5:
                    cv2.putText(image, "Pacifier", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "No Pacifier", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if active_prediction > 0.5:
                    cv2.putText(image, "Active", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Inactive", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if action_prediction == 3:  # 口鼻遮掩
                    if pacifier_prediction > 0.5:
                        send_line_message("Warning: Baby's mouth and nose are obstructed with a pacifier!")
                    else:
                        send_line_message("Warning: Baby's mouth and nose are obstructed!")
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                
                # 獲取眼睛座標點
                left_eye = np.array([[face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]] for i in range(33, 42)])
                right_eye = np.array([[face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]] for i in range(362, 371)])
                
                left_EAR = calculate_EAR(left_eye)
                right_EAR = calculate_EAR(right_eye)
                
                EAR = (left_EAR + right_EAR) / 2.0
                if EAR < 0.2:  # EAR閾值
                    cv2.putText(image, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not awake_status_notified:
                        send_line_message("Baby is awake!")
                        awake_status_notified = True
                else:
                    awake_status_notified = False
        
        # 活動偵測
        fgmask = fgbg.apply(image)
        activity_level = cv2.countNonZero(fgmask)
        if activity_level > 5000:  # 活動量閾值
            cv2.putText(image, "Baby is Active", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 畫出安全範圍
        cv2.rectangle(image, safe_zone_top_left, safe_zone_bottom_right, (0, 255, 0), 2)
        
        cv2.imshow('Baby Monitor', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
