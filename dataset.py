import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from datetime import datetime
import uuid
import os
import time
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#fields = []
#for i in range(0,42):
    #fields.append(i)
#fields.append('label')
w = csv.writer(open('data.csv','w', newline=''))
#w.writerow(fields)

new = 0
prev = 0
totframes = 0

s = "y"
dLabel = ""

def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output

def start():
    global new
    global prev
    global totframes
    global dLabel
    state = ''
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            if (state=="collecting"):
                totframes +=1
            
            ret, frame = cap.read()
            
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Flip on horizontal
            image = cv2.flip(image, 1)
            
            # Set flag
            image.flags.writeable = False
            
            # Detections
            results = hands.process(image)
            
            # Set flag to true
            image.flags.writeable = True
            
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detections
            #print(results)
            
            # Rendering results
            """ if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            ) """
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    if(state == "collecting"):
                        row = []
                        for mark in hand.landmark:
                            row.append(mark.x)
                            row.append(mark.y)
                            #row.append(mark.z)
                        row.append(dLabel)
                        w.writerow(row)
                    
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing_styles.get_default_hand_landmarks_style(),
                                            mp_drawing_styles.get_default_hand_connections_style(),
                                            )
                            
                    # Render left or right detection
                    if get_label(num, hand, results):
                        text, coord = get_label(num, hand, results)
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            # Save our image    
            #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)

            new = time.time()
            fps = 1/(new - prev)
            prev = new
            fps = int(fps)

            cv2.putText(image, "FPS:" + str(fps) + " Frames:" + str(totframes), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, "FPS:" + str(fps) + " Frames:" + str(totframes), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                totframes=0
                break
            elif cv2.waitKey(10) & 0xFF == ord('s'):
                if(state==""):
                    state="collecting"
                    print(state)

    cap.release()
    cv2.destroyAllWindows()

while (s == 'y'):
    dLabel = input("Enter label for gesture:")
    start()
    s = input("Continue?")