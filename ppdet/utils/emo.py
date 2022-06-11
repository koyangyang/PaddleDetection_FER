import paddle
import numpy as np
import cv2
from paddle.vision.models import resnet50
from paddle.vision.datasets import DatasetFolder
import paddle.nn.functional as F
import matplotlib.pylab as plt
import os


class Network(paddle.nn.Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.resnet = resnet50(pretrained=True, num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(2048, 512)
        self.linear_2 = paddle.nn.Linear(512, 256)
        self.linear_3 = paddle.nn.Linear(256, 7)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        # print('input', inputs)
        y = self.resnet(inputs)
        y = self.flatten(y)
        y = self.linear_1(y)
        y = self.linear_2(y)
        y = self.dropout(y)
        y = self.relu(y)
        y = self.linear_3(y)
        y = paddle.nn.functional.sigmoid(y)
        y = F.softmax(y)

        return y