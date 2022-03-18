
from cv2 import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Tuple, Union
#from util import (
#     draw_landmark_bbox,
#     draw_handmarks_label,
# )
#from gesture_calc import GestureCalculator
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands




FINGER_INDICES = {
    # "THUMB": [1, 2, 3, 4],
    "INDEX": [5, 6, 7, 8],
    "MIDDLE": [9, 10, 11, 12],
    "RING": [13, 14, 15, 16],
    "PINKY": [17, 18, 19, 20],
}


def sq_distance(a, b):
    return (b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2


class GestureCalculator:
    def __init__(self, landmarks: landmark_pb2.NormalizedLandmarkList):
        self.landmarks = landmarks.landmark
        self.set_finger_states()

    def process(self):
        if self.hand_is_closed():
            return "FIST"
        elif self.is_peace_sign():
            return "PEACE"
        elif self.is_pointing():
            return "POINT"
        elif self.is_rocking():
            return "ROCK!"
        return None

    def set_finger_states(self):
        self.finger_states = {"THUMB": "OPEN" if self.thumb_is_open() else "CLOSED"}
        for finger in FINGER_INDICES:
            self.finger_states[finger] = (
                "OPEN" if self.finger_is_open(finger) else "CLOSED"
            )

    def hand_is_closed(self):
        return (
            # self.finger_states["THUMB"] == "CLOSED"
            self.finger_states["INDEX"] == "CLOSED"
            and self.finger_states["MIDDLE"] == "CLOSED"
            and self.finger_states["RING"] == "CLOSED"
            and self.finger_states["PINKY"] == "CLOSED"
        )

    def is_peace_sign(self):
        return (
            # self.finger_states["THUMB"] == "CLOSED"
            self.finger_states["INDEX"] == "OPEN"
            and self.finger_states["MIDDLE"] == "OPEN"
            and self.finger_states["RING"] == "CLOSED"
            and self.finger_states["PINKY"] == "CLOSED"
        )

    def is_pointing(self):
        return (
            self.finger_states["INDEX"] == "OPEN"
            and self.finger_states["MIDDLE"] == "CLOSED"
            and self.finger_states["RING"] == "CLOSED"
            and self.finger_states["PINKY"] == "CLOSED"
        )

    def is_rocking(self):
        return (
            self.finger_states["INDEX"] == "OPEN"
            and self.finger_states["MIDDLE"] == "CLOSED"
            and self.finger_states["RING"] == "CLOSED"
            and self.finger_states["PINKY"] == "OPEN"
        )

    def finger_is_open(self, finger: str):
        """Returns:
            True if distance from TIP to WRIST is greater than distance from IP to WRIST
        """
        WRIST = self.landmarks[0]
        DIP = self.landmarks[FINGER_INDICES[finger][2]]
        TIP = self.landmarks[FINGER_INDICES[finger][3]]
        return sq_distance(WRIST, TIP) > sq_distance(WRIST, DIP)

    def thumb_is_open(self):
        # TODO: Implement a good version of this function.
        # Does not work properly in it's current form
        WRIST = self.landmarks[0]
        THUMB_CMC = self.landmarks[1]
        THUMB_MCP = self.landmarks[2]
        THUMB_IP = self.landmarks[3]
        THUMB_TIP = self.landmarks[4]
        INDEX_MCP = self.landmarks[5]
        return sq_distance(THUMB_IP, INDEX_MCP) > sq_distance(THUMB_MCP, THUMB_IP)






RED_COLOR = (0, 0, 255)


def draw_handmarks_label(
    img: np.ndarray,
    text: str,
    hand_landmarks: landmark_pb2.NormalizedLandmarkList,
    margin=20,
):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, _, _ = get_edges_in_pixels(points, img_cols, img_rows)
    pos = (x_min, (y_min - margin))
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2

    cv2.putText(img, text, pos, font_face, scale, RED_COLOR, 1, cv2.LINE_AA)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_edges_in_pixels(points: List[Tuple[int, int]], img_width: int, img_height: int):
    x_min = x_max = points[0][0]
    y_min = y_max = points[0][1]
    for x, y in points:
        x_min = x if x < x_min else x_min
        x_max = x if x > x_max else x_max
        y_min = y if y < y_min else y_min
        y_max = y if y > y_max else y_max
    return normalized_to_pixel_coordinates(
        x_min, y_min, img_width, img_height
    ) + normalized_to_pixel_coordinates(x_max, y_max, img_width, img_height)


def draw_landmark_bbox(
    img: np.ndarray, hand_landmarks: landmark_pb2.NormalizedLandmarkList
):
    img_rows, img_cols, _ = img.shape
    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    x_min, y_min, x_max, y_max = get_edges_in_pixels(points, img_cols, img_rows)
    cv2.rectangle(img, (x_min, y_max), (x_max, y_min), RED_COLOR)



# For webcam input:

hands = mp_hands.Hands(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_calc = GestureCalculator(hand_landmarks)
            gest_code = gesture_calc.process()
            if gest_code:
                draw_handmarks_label(image, gest_code, hand_landmarks)
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
hands.close()
cap.release()
