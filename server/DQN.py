import copy
import math
import pickle
import numpy as np
import scipy.misc as spm

from chainer import cuda, FunctionSet, Variable, optimizers
import chainer.functions as F

class DQN_class:
    # Hyper-Parameters
    gamma = 0.99  # Discount factor

    def __init__(self, enable_controller=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]):
        self.num_of_actions = len(enable_controller)
        self.enable_controller = enable_controller  # Default setting : "Pong"

        print "Initializing DQN..."
#	Initialization of Chainer 1.1.0 or older.
#        print "CUDA init"
#        cuda.init()

        print "Model Building"
        w = math.sqrt(2)  # MSRA scaling
        self.model = FunctionSet(
            conv1=F.Convolution2D(3,   64,  7, wscale=w, stride=2, pad=3),
            conv2_1_1=F.Convolution2D(64,   64,  1, wscale=w, stride=1),
            conv2_1_2=F.Convolution2D(64,   64,  3, wscale=w, stride=1, pad=1),
            conv2_1_3=F.Convolution2D(64,  256,  1, wscale=w, stride=1),
            conv2_1_ex=F.Convolution2D(64,  256,  1, wscale=w, stride=1),
            conv2_2_1=F.Convolution2D(256,   64,  1, wscale=w, stride=1),
            conv2_2_2=F.Convolution2D(64,   64,  3, wscale=w, stride=1, pad=1),
            conv2_2_3=F.Convolution2D(64,  256,  1, wscale=w, stride=1),
            conv2_3_1=F.Convolution2D(256,   64,  1, wscale=w, stride=1),
            conv2_3_2=F.Convolution2D(64,   64,  3, wscale=w, stride=1, pad=1),
            conv2_3_3=F.Convolution2D(64,  256,  1, wscale=w, stride=1),
            conv3_1_1=F.Convolution2D(256,  128,  1, wscale=w, stride=2),
            conv3_1_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_1_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_1_ex=F.Convolution2D(256,  512,  1, wscale=w, stride=2),
            conv3_2_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_2_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_2_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_3_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_3_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_3_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_4_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_4_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_4_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_5_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_5_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_5_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_6_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_6_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_6_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_7_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_7_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_7_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv3_8_1=F.Convolution2D(512,  128,  1, wscale=w, stride=1),
            conv3_8_2=F.Convolution2D(128,  128,  3, wscale=w, stride=1, pad=1),
            conv3_8_3=F.Convolution2D(128,  512,  1, wscale=w, stride=1),
            conv4_1_1=F.Convolution2D(512,  256,  1, wscale=w, stride=2),
            conv4_1_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_1_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_1_ex=F.Convolution2D(512,  1024,  1, wscale=w, stride=2),
            conv4_2_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_2_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_2_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_3_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_3_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_3_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_4_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_4_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_4_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_5_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_5_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_5_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_6_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_6_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_6_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_7_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_7_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_7_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_8_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_8_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_8_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_9_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_9_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_9_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_10_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_10_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_10_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_11_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_11_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_11_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_12_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_12_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_12_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_13_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_13_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_13_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_14_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_14_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_14_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_15_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_15_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_15_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_16_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_16_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_16_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_17_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_17_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_17_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_18_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_18_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_18_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_19_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_19_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_19_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_20_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_20_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_20_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_21_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_21_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_21_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_22_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_22_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_22_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_23_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_23_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_23_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_24_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_24_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_24_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_25_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_25_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_25_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_26_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_26_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_26_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_27_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_27_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_27_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_28_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_28_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_28_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_29_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_29_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_29_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_30_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_30_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_30_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_31_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_31_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_31_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_32_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_32_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_32_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_33_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_33_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_33_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_34_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_34_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_34_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_35_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_35_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_35_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv4_36_1=F.Convolution2D(1024,  256,  1, wscale=w, stride=1),
            conv4_36_2=F.Convolution2D(256,  256,  3, wscale=w, stride=1, pad=1),
            conv4_36_3=F.Convolution2D(256,  1024,  1, wscale=w, stride=1),
            conv5_1_1=F.Convolution2D(1024,  512,  1, wscale=w, stride=2),
            conv5_1_2=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv5_1_3=F.Convolution2D(512,  2048,  1, wscale=w, stride=1),
            conv5_1_ex=F.Convolution2D(1024,  2048,  1, wscale=w, stride=2),
            conv5_2_1=F.Convolution2D(2048,  512,  1, wscale=w, stride=1),
            conv5_2_2=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv5_2_3=F.Convolution2D(512,  2048,  1, wscale=w, stride=1),
            conv5_3_1=F.Convolution2D(2048,  512,  1, wscale=w, stride=1),
            conv5_3_2=F.Convolution2D(512,  512,  3, wscale=w, stride=1, pad=1),
            conv5_3_3=F.Convolution2D(512,  2048,  1, wscale=w, stride=1),
            q_value=F.Linear(2048, self.num_of_actions,
                             initialW=np.zeros((self.num_of_actions, 2048),
                                               dtype=np.float32))
        )

        self.model_target = copy.deepcopy(self.model)

        print "Initizlizing Optimizer"
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def forward(self, state, action, Reward, state_dash, episode_end):
        num_of_batch = state.shape[0]
        s = Variable(state)
        s_dash = Variable(state_dash)

        Q = self.Q_func(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.Q_func_target(s_dash)  # Q(s',*)
        tmp = list(map(np.max, tmp.data.get()))  # max_a Q(s',a)
        max_Q_dash = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(Q.data.get(), dtype=np.float32)

        for i in xrange(num_of_batch):
            if not episode_end[i][0]:
                tmp_ = np.sign(Reward[i]) + self.gamma * max_Q_dash[i]
            else:
                tmp_ = np.sign(Reward[i])

            action_index = self.action_to_index(action[i])
            target[i, action_index] = tmp_

        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - Q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((num_of_batch, self.num_of_actions), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss, Q

    def Q_func(self, state):
        h = F.relu(self.model.conv1(state))
        h = F.max_pooling_2d(h, 3, stride=2)

        h_rem = self.model.conv2_1_ex(h)
        h = F.relu(self.model.conv2_1_1(h))
        h = F.relu(self.model.conv2_1_2(h))
        h = self.model.conv2_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv2_2_1(h))
        h = F.relu(self.model.conv2_2_2(h))
        h = self.model.conv2_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv2_3_1(h))
        h = F.relu(self.model.conv2_3_2(h))
        h = self.model.conv2_3_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model.conv3_1_ex(h)
        h = F.relu(self.model.conv3_1_1(h))
        h = F.relu(self.model.conv3_1_2(h))
        h = self.model.conv3_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_2_1(h))
        h = F.relu(self.model.conv3_2_2(h))
        h = self.model.conv3_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_3_1(h))
        h = F.relu(self.model.conv3_3_2(h))
        h = self.model.conv3_3_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_4_1(h))
        h = F.relu(self.model.conv3_4_2(h))
        h = self.model.conv3_4_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_5_1(h))
        h = F.relu(self.model.conv3_5_2(h))
        h = self.model.conv3_5_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_6_1(h))
        h = F.relu(self.model.conv3_6_2(h))
        h = self.model.conv3_6_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_7_1(h))
        h = F.relu(self.model.conv3_7_2(h))
        h = self.model.conv3_7_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv3_8_1(h))
        h = F.relu(self.model.conv3_8_2(h))
        h = self.model.conv3_8_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model.conv4_1_ex(h)
        h = F.relu(self.model.conv4_1_1(h))
        h = F.relu(self.model.conv4_1_2(h))
        h = self.model.conv4_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_2_1(h))
        h = F.relu(self.model.conv4_2_2(h))
        h = self.model.conv4_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_3_1(h))
        h = F.relu(self.model.conv4_3_2(h))
        h = self.model.conv4_3_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_4_1(h))
        h = F.relu(self.model.conv4_4_2(h))
        h = self.model.conv4_4_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_5_1(h))
        h = F.relu(self.model.conv4_5_2(h))
        h = self.model.conv4_5_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_6_1(h))
        h = F.relu(self.model.conv4_6_2(h))
        h = self.model.conv4_6_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_7_1(h))
        h = F.relu(self.model.conv4_7_2(h))
        h = self.model.conv4_7_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_8_1(h))
        h = F.relu(self.model.conv4_8_2(h))
        h = self.model.conv4_8_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_9_1(h))
        h = F.relu(self.model.conv4_9_2(h))
        h = self.model.conv4_9_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_10_1(h))
        h = F.relu(self.model.conv4_10_2(h))
        h = self.model.conv4_10_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_11_1(h))
        h = F.relu(self.model.conv4_11_2(h))
        h = self.model.conv4_11_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_12_1(h))
        h = F.relu(self.model.conv4_12_2(h))
        h = self.model.conv4_12_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_13_1(h))
        h = F.relu(self.model.conv4_13_2(h))
        h = self.model.conv4_13_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_14_1(h))
        h = F.relu(self.model.conv4_14_2(h))
        h = self.model.conv4_14_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_15_1(h))
        h = F.relu(self.model.conv4_15_2(h))
        h = self.model.conv4_15_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_16_1(h))
        h = F.relu(self.model.conv4_16_2(h))
        h = self.model.conv4_16_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_17_1(h))
        h = F.relu(self.model.conv4_17_2(h))
        h = self.model.conv4_17_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_18_1(h))
        h = F.relu(self.model.conv4_18_2(h))
        h = self.model.conv4_18_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_19_1(h))
        h = F.relu(self.model.conv4_19_2(h))
        h = self.model.conv4_19_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_20_1(h))
        h = F.relu(self.model.conv4_20_2(h))
        h = self.model.conv4_20_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_21_1(h))
        h = F.relu(self.model.conv4_21_2(h))
        h = self.model.conv4_21_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_22_1(h))
        h = F.relu(self.model.conv4_22_2(h))
        h = self.model.conv4_22_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_23_1(h))
        h = F.relu(self.model.conv4_23_2(h))
        h = self.model.conv4_23_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_24_1(h))
        h = F.relu(self.model.conv4_24_2(h))
        h = self.model.conv4_24_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_25_1(h))
        h = F.relu(self.model.conv4_25_2(h))
        h = self.model.conv4_25_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_26_1(h))
        h = F.relu(self.model.conv4_26_2(h))
        h = self.model.conv4_26_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_27_1(h))
        h = F.relu(self.model.conv4_27_2(h))
        h = self.model.conv4_27_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_28_1(h))
        h = F.relu(self.model.conv4_28_2(h))
        h = self.model.conv4_28_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_29_1(h))
        h = F.relu(self.model.conv4_29_2(h))
        h = self.model.conv4_29_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_30_1(h))
        h = F.relu(self.model.conv4_30_2(h))
        h = self.model.conv4_30_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_31_1(h))
        h = F.relu(self.model.conv4_31_2(h))
        h = self.model.conv4_31_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_32_1(h))
        h = F.relu(self.model.conv4_32_2(h))
        h = self.model.conv4_32_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_33_1(h))
        h = F.relu(self.model.conv4_33_2(h))
        h = self.model.conv4_33_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_34_1(h))
        h = F.relu(self.model.conv4_34_2(h))
        h = self.model.conv4_34_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_35_1(h))
        h = F.relu(self.model.conv4_35_2(h))
        h = self.model.conv4_35_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv4_36_1(h))
        h = F.relu(self.model.conv4_36_2(h))
        h = self.model.conv4_36_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model.conv5_1_ex(h)
        h = F.relu(self.model.conv5_1_1(h))
        h = F.relu(self.model.conv5_1_2(h))
        h = self.model.conv5_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv5_2_1(h))
        h = F.relu(self.model.conv5_2_2(h))
        h = self.model.conv5_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model.conv5_3_1(h))
        h = F.relu(self.model.conv5_3_2(h))
        h = self.model.conv5_3_3(h)
        h = F.relu(h + h_rem)

        h = F.average_pooling_2d(h, 7)
        Q = self.model.q_value(h)
        return Q

    def Q_func_target(self, state):
        h = F.relu(self.model_target.conv1(state))
        h = F.max_pooling_2d(h, 3, stride=2)

        h_rem = self.model_target.conv2_1_ex(h)
        h = F.relu(self.model_target.conv2_1_1(h))
        h = F.relu(self.model_target.conv2_1_2(h))
        h = self.model_target.conv2_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv2_2_1(h))
        h = F.relu(self.model_target.conv2_2_2(h))
        h = self.model_target.conv2_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv2_3_1(h))
        h = F.relu(self.model_target.conv2_3_2(h))
        h = self.model_target.conv2_3_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model_target.conv3_1_ex(h)
        h = F.relu(self.model_target.conv3_1_1(h))
        h = F.relu(self.model_target.conv3_1_2(h))
        h = self.model_target.conv3_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_2_1(h))
        h = F.relu(self.model_target.conv3_2_2(h))
        h = self.model_target.conv3_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_3_1(h))
        h = F.relu(self.model_target.conv3_3_2(h))
        h = self.model_target.conv3_3_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_4_1(h))
        h = F.relu(self.model_target.conv3_4_2(h))
        h = self.model_target.conv3_4_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_5_1(h))
        h = F.relu(self.model_target.conv3_5_2(h))
        h = self.model_target.conv3_5_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_6_1(h))
        h = F.relu(self.model_target.conv3_6_2(h))
        h = self.model_target.conv3_6_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_7_1(h))
        h = F.relu(self.model_target.conv3_7_2(h))
        h = self.model_target.conv3_7_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv3_8_1(h))
        h = F.relu(self.model_target.conv3_8_2(h))
        h = self.model_target.conv3_8_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model_target.conv4_1_ex(h)
        h = F.relu(self.model_target.conv4_1_1(h))
        h = F.relu(self.model_target.conv4_1_2(h))
        h = self.model_target.conv4_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_2_1(h))
        h = F.relu(self.model_target.conv4_2_2(h))
        h = self.model_target.conv4_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_3_1(h))
        h = F.relu(self.model_target.conv4_3_2(h))
        h = self.model_target.conv4_3_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_4_1(h))
        h = F.relu(self.model_target.conv4_4_2(h))
        h = self.model_target.conv4_4_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_5_1(h))
        h = F.relu(self.model_target.conv4_5_2(h))
        h = self.model_target.conv4_5_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_6_1(h))
        h = F.relu(self.model_target.conv4_6_2(h))
        h = self.model_target.conv4_6_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_7_1(h))
        h = F.relu(self.model_target.conv4_7_2(h))
        h = self.model_target.conv4_7_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_8_1(h))
        h = F.relu(self.model_target.conv4_8_2(h))
        h = self.model_target.conv4_8_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_9_1(h))
        h = F.relu(self.model_target.conv4_9_2(h))
        h = self.model_target.conv4_9_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_10_1(h))
        h = F.relu(self.model_target.conv4_10_2(h))
        h = self.model_target.conv4_10_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_11_1(h))
        h = F.relu(self.model_target.conv4_11_2(h))
        h = self.model_target.conv4_11_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_12_1(h))
        h = F.relu(self.model_target.conv4_12_2(h))
        h = self.model_target.conv4_12_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_13_1(h))
        h = F.relu(self.model_target.conv4_13_2(h))
        h = self.model_target.conv4_13_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_14_1(h))
        h = F.relu(self.model_target.conv4_14_2(h))
        h = self.model_target.conv4_14_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_15_1(h))
        h = F.relu(self.model_target.conv4_15_2(h))
        h = self.model_target.conv4_15_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_16_1(h))
        h = F.relu(self.model_target.conv4_16_2(h))
        h = self.model_target.conv4_16_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_17_1(h))
        h = F.relu(self.model_target.conv4_17_2(h))
        h = self.model_target.conv4_17_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_18_1(h))
        h = F.relu(self.model_target.conv4_18_2(h))
        h = self.model_target.conv4_18_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_19_1(h))
        h = F.relu(self.model_target.conv4_19_2(h))
        h = self.model_target.conv4_19_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_20_1(h))
        h = F.relu(self.model_target.conv4_20_2(h))
        h = self.model_target.conv4_20_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_21_1(h))
        h = F.relu(self.model_target.conv4_21_2(h))
        h = self.model_target.conv4_21_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_22_1(h))
        h = F.relu(self.model_target.conv4_22_2(h))
        h = self.model_target.conv4_22_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_23_1(h))
        h = F.relu(self.model_target.conv4_23_2(h))
        h = self.model_target.conv4_23_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_24_1(h))
        h = F.relu(self.model_target.conv4_24_2(h))
        h = self.model_target.conv4_24_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_25_1(h))
        h = F.relu(self.model_target.conv4_25_2(h))
        h = self.model_target.conv4_25_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_26_1(h))
        h = F.relu(self.model_target.conv4_26_2(h))
        h = self.model_target.conv4_26_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_27_1(h))
        h = F.relu(self.model_target.conv4_27_2(h))
        h = self.model_target.conv4_27_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_28_1(h))
        h = F.relu(self.model_target.conv4_28_2(h))
        h = self.model_target.conv4_28_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_29_1(h))
        h = F.relu(self.model_target.conv4_29_2(h))
        h = self.model_target.conv4_29_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_30_1(h))
        h = F.relu(self.model_target.conv4_30_2(h))
        h = self.model_target.conv4_30_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_31_1(h))
        h = F.relu(self.model_target.conv4_31_2(h))
        h = self.model_target.conv4_31_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_32_1(h))
        h = F.relu(self.model_target.conv4_32_2(h))
        h = self.model_target.conv4_32_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_33_1(h))
        h = F.relu(self.model_target.conv4_33_2(h))
        h = self.model_target.conv4_33_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_34_1(h))
        h = F.relu(self.model_target.conv4_34_2(h))
        h = self.model_target.conv4_34_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_35_1(h))
        h = F.relu(self.model_target.conv4_35_2(h))
        h = self.model_target.conv4_35_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv4_36_1(h))
        h = F.relu(self.model_target.conv4_36_2(h))
        h = self.model_target.conv4_36_3(h)
        h = F.relu(h + h_rem)

        h_rem = self.model_target.conv5_1_ex(h)
        h = F.relu(self.model_target.conv5_1_1(h))
        h = F.relu(self.model_target.conv5_1_2(h))
        h = self.model_target.conv5_1_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv5_2_1(h))
        h = F.relu(self.model_target.conv5_2_2(h))
        h = self.model_target.conv5_2_3(h)
        h = F.relu(h + h_rem)
        h_rem = h
        h = F.relu(self.model_target.conv5_3_1(h))
        h = F.relu(self.model_target.conv5_3_2(h))
        h = self.model_target.conv5_3_3(h)
        h = F.relu(h + h_rem)

        h = F.average_pooling_2d(h, 7)
        Q = self.model_target.q_value(h)
        return Q

    def e_greedy(self, state, epsilon):
        s = Variable(state)
        Q = self.Q_func(s)
        Q = Q.data

        if np.random.rand() < epsilon:
            index_action = np.random.randint(0, self.num_of_actions)
            print "RANDOM"
        else:
            index_action = np.argmax(Q.get())
            print "GREEDY"
        return self.index_to_action(index_action)

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.enable_controller[index_of_action]

    def action_to_index(self, action):
        return self.enable_controller.index(action)