import math
import chainer
import chainer.functions as F
import chainer.links as L
import cv2
import cupy as cp
import numpy as np

from chainer import Variable

class deathcheck(chainer.FunctionSet):

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(resnet, self).__init__(
            fc1=F.Linear(3*98*30,10000),
            fc2=F.Linear(10000,10000),
            fc3=F.Linear(10000,2),
        )

    def forward(self, x_data, y_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        t = chainer.Variable(y_data, volatile=not train)

        h = F.relu(F.dropout(self.fc1(x),train=train))
        h = F.relu(F.dropout(self.fc2(h),train=train))
        h = F.relu(self.fc3(h))

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def predict(self, x_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        
        h = F.relu(F.dropout(self.fc1(x),train=train))
        h = F.relu(F.dropout(self.fc2(h),train=train))
        h = F.relu(self.fc3(h))

        return F.softmax(h)
