import cv2 as cv
import numpy as np
import mediapipe as mp
from utils import *
from UIUtils import *
from UIService import UIService
import time

win_name = "Evaluate on Air"

camera = cv.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
ui_service = UIService(1)

flag, frame = camera.read()

print(frame.shape)

cv.namedWindow(win_name)
cv.resizeWindow(win_name, frame.shape[1], frame.shape[0])

line_xy = None

mode = -1

canvas = buildCanvas(frame.shape)

while flag == True:
    hands_image = np.zeros(frame.shape)

    cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = hands.process(frame)

    if res.multi_hand_landmarks:
        heights = []
        draw_xy = ()
        points = []
        for landmarks in res.multi_hand_landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                x, y = landmark.x, landmark.y
                height, width = frame.shape[0], frame.shape[1]
                px, py = (width - int(x * width), int(height * y))
                heights.append(int(height * y))
                points.append((px, py))
                if idx == 8:
                    draw_xy = (px, py)
                cv.circle(hands_image, (px, py), 3, (255, 0, 255), cv.FILLED)
        max_idx = findMin(heights)
        if max_idx == 8 or heights[8] < heights[12]:
            mode = 1
            print("Mode : Draw or Select")
            if pointDist(draw_xy, (575, 80)) <= 50:
                canvas = buildCanvas(frame.shape)
            elif buttonPressed(draw_xy) != 0:
                code = buttonPressed(draw_xy)
                result = ui_service.performOp(canvas, code)
                print("Result : " + str(result))
                canvas = writeResult(str(result), canvas)
                # canvas = np.where(res_image, res_image, canvas)
                time.sleep(0.5)
            elif checkIfWithinRec(draw_xy):
                if line_xy is None:
                    cv.line(canvas, draw_xy, draw_xy, (0, 0, 255), 5)
                else:
                    cv.line(canvas, line_xy, draw_xy, (0, 0, 255), 5)
                line_xy = draw_xy
                
        else: 
            mode = 0
            print("Mode: Cursor")
            cv.circle(hands_image, (points[8][0], points[8][1]), 7, (255, 0, 0), cv.FILLED)
            cv.circle(hands_image, (points[12][0], points[12][1]), 7, (255, 0, 0), cv.FILLED)
            line_xy = None

    cv.imshow("frame", frame)

    cv.imshow(win_name, np.where(hands_image, hands_image, canvas))

    flag, frame = camera.read()

    key = cv.waitKey(1)
    if key == ord('q'):
        break

camera.release()
# cv.waitKey(0)
cv.destroyAllWindows()
