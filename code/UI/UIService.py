import cv2 as cv
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.ModelServices import *
from utils import *

class UIService:
    def __init__(self, code):
        self.model_services = ModelServices(code)

    def performOp(self, canvas, op):
        operand1 = []
        for i in range(51, 250):
            row = []
            for j in range(151, 350):
                row.append(canvas[j][i])
            operand1.append(row)

        operand1 = np.array(operand1)
        operand1 = cv.flip(operand1, 0)
        operand1 = cv.rotate(operand1, cv.ROTATE_90_CLOCKWISE)
        cv.imshow("op1", operand1)

        operand2 = []
        for i in range(401, 600):
            row = []
            for j in range(151, 350):
                row.append(canvas[j][i])
            operand2.append(row)

        operand2 = np.array(operand2)
        operand2 = cv.flip(operand2, 0)
        operand2 = cv.rotate(operand2, cv.ROTATE_90_CLOCKWISE)
        
        cv.imshow("op2", operand2)
        cv.imshow("canvas", canvas)

        values = self.model_services.predict(operand1)
        op1 = findMax(values)
        print(values)
        print(op1)

        values = self.model_services.predict(operand2)
        op2 = findMax(values)
        print(values)
        print(op2)

        return getResult(op, op1, op2)
    
    