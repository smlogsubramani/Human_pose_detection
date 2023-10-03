import cv2
import mediapipe as mp
import numpy as np
# import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  
bicep_curl_count = 0
up_phase = True 
elbow_flexion_threshold = 90 
elbow_extension_threshold = 150


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if(angle > 180.0):
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Store result
        results = pose.process(image)
        # Recolored image back to normal
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            rangle = calculate_angle(rshoulder, relbow, rwrist)

            if angle < elbow_flexion_threshold and rangle < elbow_flexion_threshold and up_phase:
                up_phase = False
            elif angle > elbow_extension_threshold and rangle > elbow_extension_threshold and not up_phase:
                up_phase = True  
                bicep_curl_count += 1  

            

            # Visualize
            angle_text = f"left elbow Angle: {angle:.2f} degrees" 
            rangle_text = f"right elbow Angle: {rangle:.2f} degrees"  
            count_text = f"Bicep Curls: {bicep_curl_count}"
            cv2.putText(image, count_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(image, angle_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(image, rangle_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

            # Print coordinates
            # print(f"Shoulder: {shoulder}, Elbow: {elbow}, Wrist: {wrist}")
            print(f"left-elbow-angle : {angle_text} , right-elbow-angle:{rangle_text}")
        except:
            pass

        # Render detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('l'):
            break

cap.release()
cv2.destroyAllWindows()