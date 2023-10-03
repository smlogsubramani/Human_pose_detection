import cv2
import mediapipe as mp
import numpy as np 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose  

#https://github.com/google/mediapipe/tree/master/docs/solutions - branch under solution and pose .

#funciton defintion

def calculate_angle(a,b,c):
    a=np.array(a);
    b=np.array(b);
    c=np.array(c);
    radians = np.arctan(c[1]-b[1] , c[0]-b[0]) - np.arctan(a[1]-b[1] , a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if(angle > 180.0):
        angle = 360-angle
    return angle  
    # ba = a - b
    # bc = c - b

    # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # angle_rad = np.arccos(cosine_angle)
    # angle = np.degrees(angle_rad)

    # return angle



#creating the vedio feed from the webcam
cap = cv2.VideoCapture(0)
# setting up media pipe instance
with mp_pose.Pose(min_detection_confidence = 0.5,min_tracking_confidence = 0.5) as pose:

    while cap.isOpened():
        ret , frame = cap.read()

        #detect image

        #recolor image
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        #store result 
        results = pose.process(image)
        #recolored image back to normal
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


        try:
            landmarks = results.landmarks.landmark

            #get coordinated 
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            #calculate angle
            angle = calculate_angle(shoulder,elbow,wrist)

            #visualize
            cv2.putText(image,str(angle),tuple(np.multiply(elbow,[640,480]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
            print(shoulder ,elbow ,wrist)
        except:
            pass

        #render detection
        mp_drawing.draw_landmarks(image , results.pose_landmarks,mp_pose.POSE_CONNECTIONS , mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed' ,image)

        if cv2.waitKey(10) & 0xFF == ord('l'): 
            break
cap.release()
cv2.destroyAllWindows()

