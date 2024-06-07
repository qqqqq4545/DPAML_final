import cv2
import mediapipe as mp
import numpy as np
import requests
import collections
import tensorflow as tf

# 加载 TensorFlow Lite 模型
interpreter = tf.lite.Interpreter(model_path='/mnt/data/InceptionResNetV2.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 定义categories变量
categories = ['normal', 'cover', 'pacifier']

# 初始化MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 设置安全范围坐标
safe_zone_top_left = (50, 50)
safe_zone_bottom_right = (550, 450)

# LINE Notify Token
line_token = 'YOUR_LINE_NOTIFY_TOKEN'

# 发送LINE通知
def send_line_message(message):
    headers = {
        "Authorization": "Bearer " + line_token,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {'message': message}
    response = requests.post("https://notify-api.line.me/api/notify", headers=headers, params=payload)
    return response

# EAR计算
def calculate_EAR(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# 打开摄像机
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

# 初始化时间序列缓冲区
sequence_length = 1
frame_buffer = collections.deque(maxlen=sequence_length)
awake_status_notified = False
inside_status_notified = True  # 初始状态假设在安全范围内

try:
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error: Failed to read image from camera.")
                break
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 姿势检测
            pose_results = pose.process(image_rgb)
            face_results = face_mesh.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            if pose_results.pose_landmarks:
                print("Pose landmarks detected")
                # 绘制姿势标记
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=pose_results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
                
                # 获取身体中心点
                body_center_x = int(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.shape[1])
                body_center_y = int(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.shape[0])
                
                # 检查是否在安全范围内
                if (safe_zone_top_left[0] <= body_center_x <= safe_zone_bottom_right[0]) and \
                   (safe_zone_top_left[1] <= body_center_y <= safe_zone_bottom_right[1]):
                    cv2.putText(image, "Inside", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    inside_status_notified = False
                else:
                    cv2.putText(image, "Outside", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not inside_status_notified:
                        send_line_message("Warning: Baby is outside the safe zone!")
                        inside_status_notified = True
                
                # 创建ROI并调整大小
                roi = image[safe_zone_top_left[1]:safe_zone_bottom_right[1], safe_zone_top_left[0]:safe_zone_bottom_right[0]]
                roi_resized = cv2.resize(roi, (224, 224))
                roi_resized = roi_resized / 255.0
                frame_buffer.append(roi_resized)
                
                if len(frame_buffer) == sequence_length:
                    sequence = np.array(frame_buffer, dtype=np.float32)
                    sequence = np.expand_dims(sequence, axis=0)
                    interpreter.set_tensor(input_details[0]['index'], sequence)
                    interpreter.invoke()
                    predictions = interpreter.get_tensor(output_details[0]['index'])
                    
                    prediction = np.argmax(predictions[0])
                    category = categories[prediction]
                    
                    if category == 'cover':
                        cv2.putText(image, "Cover", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        send_line_message("Warning: Baby's mouth and nose are covered!")
                    elif category == 'pacifier':
                        cv2.putText(image, "Pacifier", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 不发送消息
                    elif category == 'normal':
                        cv2.putText(image, "Normal", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 不发送消息
            else:
                print("No pose landmarks detected")
            
            if face_results.multi_face_landmarks:
                print("Face landmarks detected")
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
                    
                    # 获取眼睛坐标点
                    left_eye = np.array([[face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]] for i in range(33, 42)])
                    right_eye = np.array([[face_landmarks.landmark[i].x * image.shape[1], face_landmarks.landmark[i].y * image.shape[0]] for i in range(362, 371)])
                    
                    left_EAR = calculate_EAR(left_eye)
                    right_EAR = calculate_EAR(right_eye)
                    
                    EAR = (left_EAR + right_EAR) / 2.0
                    if EAR < 0.2:  # EAR阈值
                        cv2.putText(image, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        if not awake_status_notified:
                            send_line_message("Baby is awake!")
                            awake_status_notified = True
                    else:
                        awake_status_notified = False
            else:
                print("No face landmarks detected")
            
            # 活动检测
            fgmask = fgbg.apply(image)
            activity_level = cv2.countNonZero(fgmask)
            if activity_level > 10000:  # 活动量阈值
                cv2.putText(image, "Baby is Active", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 画出安全范围
            cv2.rectangle(image, safe_zone_top_left, safe_zone_bottom_right, (0, 255, 0), 2)
            
            # 确保图像显示
            print("Attempting to display image")
            cv2.imshow('Baby Monitor', image)
            print("Image displayed")
            if cv2.waitKey(5) & 0xFF == 27:
                print("Esc key pressed, exiting loop")
                break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and destroyed all windows")
