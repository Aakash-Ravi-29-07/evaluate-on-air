import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Model.ModelOps import *
import cv2 as cv
import torch
import math
import matplotlib.pyplot as plt 

class ModelServices:
    def __init__(self, mode):
        self.modelOps = ModelOps(64, 1000, 0.01, 0.5, (5, 5), (2, 2), 20)
        if mode == 0:
            self.modelOps.createNewModel()
            self.modelOps.train()
            self.modelOps.saveModel("D:\Computer Vision\evaluate-on-air\myModel")
            self.modelOps.test()
        else:
            self.modelOps.loadModel("D:\Computer Vision\evaluate-on-air\myModel")
            self.modelOps.test()

    def predict(self, image):
        image = cv.resize(image, (28, 28))
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # print(gray_image.shape)
        gray_image.reshape((1, 28, 28))
        tensor = torch.Tensor([gray_image])
        plt.imshow(gray_image)
        print(tensor.shape)
        result = self.modelOps.predict(tensor).tolist()[0]
        exp_values = [math.exp(x) for x in result]

        return exp_values

    