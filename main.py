from collections import deque
import imutils
import argparse
import numpy as np
import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=20,
	help="max buffer size")
args = vars(ap.parse_args())

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    (image_height, image_width, _) = img.shape

    if results.multi_hand_landmarks:
        for (idx, hand_landmarks) in \
                enumerate(results.multi_hand_landmarks):

                thumbCoords = \
                    (hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x
                     * image_width,
                     hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y
                     * image_height)
                indexCoords = \
                    (hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x
                     * image_width,
                     hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y
                     * image_height)
                middleCoords = \
                    (hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x
                     * image_width,
                     hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y
                     * image_height)
                ringCoords = \
                    (hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].x
                     * image_width,
                     hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP].y
                     * image_height)
                pinkyCoords = \
                    (hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x
                     * image_width,
                     hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y
                     * image_height)


                cv2.circle(img, (int(thumbCoords[0]),
                           int(thumbCoords[1])), 6, (255,255,255), cv2.FILLED)
                cv2.circle(img, (int(indexCoords[0]),
                           int(indexCoords[1])), 6, (255,255,255), cv2.FILLED)
                cv2.circle(img, (int(middleCoords[0]),
                           int(middleCoords[1])), 6, (255,255,255), cv2.FILLED)
                cv2.circle(img, (int(ringCoords[0]),
                           int(ringCoords[1])), 6, (255,255,255), cv2.FILLED)
                cv2.circle(img, (int(pinkyCoords[0]),
                           int(pinkyCoords[1])), 6, (255,255,255), cv2.FILLED)
        # mpDraw.draw_landmarks(img, center_coordinates1)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # flip and blur the image
    img = cv2.flip(img, 1)
    img = imutils.resize(img, width = 1000, height = 1000)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(5) == ord('q'):
    		break