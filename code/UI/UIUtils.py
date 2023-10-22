import numpy as np
import cv2 as cv

def buildCanvas(shape):
    canvas = np.zeros(shape, np.uint8)
    cv.circle(canvas, (575, 80), 40, (125, 130, 200), cv.FILLED)
    cv.putText(canvas, "Clear", (540, 80), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv.rectangle(canvas, (50, 150), (250, 350), (0, 255, 255))
    cv.rectangle(canvas, (400, 150), (600, 350), (0, 255, 255))
    cv.rectangle(canvas, (50, 400), (100, 450), (255, 255, 255), cv.FILLED)
    cv.putText(canvas, "+", (70, 430), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv.rectangle(canvas, (150, 400), (200, 450), (255, 255, 255), cv.FILLED)
    cv.putText(canvas, "-", (170, 430), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv.rectangle(canvas, (250, 400), (300, 450), (255, 255, 255), cv.FILLED)
    cv.putText(canvas, "x", (270, 430), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv.rectangle(canvas, (350, 400), (400, 450), (255, 255, 255), cv.FILLED)
    cv.putText(canvas, "/", (370, 430), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    cv.rectangle(canvas, (275, 200), (375, 300), (255, 0, 255), cv.FILLED)
    return canvas

def buttonPressed(xy):
    if 50 < xy[0] < 100 and 400 < xy[1] < 450: # add
        return 1
    elif 150 < xy[0] < 200 and 400 < xy[1] < 450:
        return 2
    elif 250 < xy[0] < 300 and 400 < xy[1] < 450:
        return 3
    elif 350 < xy[0] < 400 and 400 < xy[1] < 450:
        return 4
    else:
        return 0
    
def checkIfWithinRec(xy):
    if 50 < xy[0] < 250 and 150 < xy[1] < 350:
        return True
    elif 400 < xy[0] < 600 and 150 < xy[1] < 350:
        return True
    return False

def writeResult(result, canvas):
       cv.rectangle(canvas, (275, 200), (375, 300), (255, 0, 255), cv.FILLED)
       cv.putText(canvas, result, (300, 235), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2) 
       return canvas

