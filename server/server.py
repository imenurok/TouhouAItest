# -*- coding:utf-8 -*-
import SocketServer
import socket
import cv2
import os
import numpy as np
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
import subprocess

import numpy as np
from chainer.cuda import cupy as cp
from PIL import Image

import math
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
import random
import six
import six.moves.cPickle as pickle
from six.moves import queue

import DeathCheck
import DQN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import pylab

parser = argparse.ArgumentParser(
    description='Image inspection using chainer')
parser.add_argument('--model','-m',default='model', help='Path to model file')
parser.add_argument('--model2','-m2',default='model2', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--mean2', default='mean2.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--gpu2', '-g2', default=1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

categories = np.loadtxt("labels.txt", str, delimiter="\t")

f=open(args.mean,'rb')
mean_image = pickle.load(f)
f.close()

f=open(args.mean2,'rb')
mean_image2 = pickle.load(f)
f.close()

def img_pass_tuple(state,action):
    _state=""
    _action=cp.zeros(1, dtype=cp.uint8)
    _state=state
    _action[0]=cp.asarray(action)
    return (_state,_action)

def img_tuple(state,action,Reward,state_dash,episode_end):
    _state=""
    _action=cp.zeros(1, dtype=cp.uint8)
    _Reward=np.zeros((1, 1), dtype=np.float32)
    _state_dash=""
    _episode_end=np.zeros((1, 1), dtype=np.bool)
    _state=state
    _action[0]=cp.asarray(action)
    _Reward[0][0]=Reward
    _state_dash=state_dash
    _episode_end[0][0]=episode_end
    return (_state,_action,_Reward,_state_dash,_episode_end)

def load_img(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1).astype(np.float32)
    image -= mean_image2
    image /= 255
    return cp.asarray(image.reshape(1,3,224,224))

size=224

class TCPHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        save_img=0
        
        f=open(args.model,'rb')
        model = pickle.load(f)
        f.close()
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        
        if os.path.isfile(args.model2):
            f=open(args.model2,'rb')
            model2 = DQN.DQN_class()
            model2.model = pickle.load(f)
            model2.target_model_update()
            f.close()
           # cuda.get_device(args.gpu2).use()
            model2.model.to_gpu()
            model2.model_target.to_gpu()
        else:
            model2 = DQN.DQN_class()
           # cuda.get_device(args.gpu2).use()
            model2.model.to_gpu()
            model2.model_target.to_gpu()
        
        que_img = []
        que_img_pass = []
        #train = open('train.txt','w')
        episodeCounter=0
        print "episode "+str(episodeCounter+1)
        double_error=0
        linelist=["python compute_mean.py train.txt","python train_imagenet.py -g 0 -B 16 -b 1 -E 20 train.txt test.txt 2>&1 | tee log"]
        eps=1
        base_time=time.time()
        time_bool=True
        try:
            while True:
                self.data = self.request.recv(1024).strip()
                if time_bool:
                    time_bool=False
                    base_time=base_time-(base_time-time.time())
                self.request.send("receive")
                buf=''
                recvlen=0
                while recvlen<int(self.data): 
                    receivedstr=self.request.recv(1024*1024)
                    recvlen+=len(receivedstr)
                    buf +=receivedstr
                img=np.fromstring(buf,dtype='uint8').reshape(740-290,670-284,3)
                img_check=img[464-290:494-290,284-284:382-284,:].astype(np.float32)
                _img_check = img_check.transpose(2,0,1) - mean_image
                #rsvmsg=cv2.imdecode(rsvmsg,1)
                height, width, depth = img.shape
                new_height = size
                new_width = size
                if height > width:
                    new_width = size * width / height
                else:
                    new_height = size * height / width
                crop_height_start = ( size - new_height ) / 2
                crop_height_end = crop_height_start + new_height
                crop_width_start = ( size - new_width) / 2
                crop_width_end = crop_width_start + new_width
                resized_img = cv2.resize(img, (new_width, new_height))
                cropped_img = np.zeros((size,size,3),np.uint8)
                #cropped_img.fill(255) #white ver
                cropped_img[crop_height_start:crop_height_end,crop_width_start:crop_width_end] = resized_img
                _cropped_img = cropped_img.astype(np.float32).transpose(2,0,1)
                _cropped_img -= mean_image2
                _cropped_img /= 255
                state_dash = cp.ndarray((1, 3, size, size), dtype=cp.float32)
                state_check = cp.ndarray((1, 3, 30, 98), dtype=cp.float32)
                state_dash[0]=cp.asarray(_cropped_img)
                state_check[0]=cp.asarray(_img_check)
                action_dash = model2.e_greedy(state_dash, eps)
                _name=np.argmax(model.predict(state_check,train=False).data.get())
                print action_dash
                state_dash_path="save/img"+str(save_img).zfill(7)+".jpg"
                cv2.imwrite(state_dash_path,cropped_img)
                print "saved "+state_dash_path
                save_img+=1
                if len(que_img_pass)>0:
                    state,action=que_img_pass.pop(0)
                    Reward=(time.time()-base_time)*(1.0/10**3)
                    episode_end=False
                    if _name==-1:
                        Reward=-1
                        episode_end=True
                    que_img.append(img_tuple(state,action,Reward,state_dash_path,episode_end))
                    print Reward
                if _name==1:
                    self.request.send(str(-2))
                    self.request.recv(1024)
                    if episodeCounter==19:
                        while len(que_img)>100000:
                            que_img.pop(0)
                        eps=1
                        DQN_batch_size=1
                        _state=cp.zeros((DQN_batch_size, 3, 224, 224),dtype=cp.float32)
                        _action=cp.zeros(DQN_batch_size, dtype=cp.uint8)
                        _Reward=np.zeros((DQN_batch_size, 1), dtype=np.float32)
                        _state_dash=cp.zeros((DQN_batch_size, 3, 224, 224), dtype=cp.float32)
                        _episode_end=np.zeros((DQN_batch_size, 1), dtype=np.bool)
                        for epoch in range(5):
                            print "epoch"+str(epoch+1)
                            counter=0
                            perm = np.random.permutation(len(que_img))
                            for idx in perm:
                                s_replay,a_replay,r_replay,s_dash_replay,episode_end_replay=que_img[idx]
                                _state[counter:counter+1]=load_img(s_replay)
                                _action[counter:counter+1]=a_replay
                                _Reward[counter:counter+1]=r_replay
                                _state_dash[counter:counter+1]=load_img(s_dash_replay)
                                _episode_end[counter:counter+1]=episode_end_replay
                                if counter==DQN_batch_size-1:
                                    counter=0
                                    model2.optimizer.zero_grads()
                                    loss, _ = model2.forward(_state,_action,_Reward,_state_dash,_episode_end)
                                    loss.backward()
                                    model2.optimizer.update()
                                    del loss
                                else:
                                    counter+=1
                                del s_replay,a_replay,r_replay,s_dash_replay,episode_end_replay
                        episodeCounter=0
                        pickle.dump(model2.model, open(args.model2, 'wb'), -1)
                        model2.target_model_update()
                        #model2.model_target.to_gpu()
                    else:
                        episodeCounter+=1
                    base_time=time.time()
                    time_bool=True
                    print "episode "+str(episodeCounter+1)
                    self.request.send("OK")
                else:
                    que_img_pass.append(img_pass_tuple(state_dash_path,action_dash))
                    self.request.send(str(action_dash))
                    eps -= 5.0/10**4
                    print eps
                    if eps < 0.1:
                        eps = 0.1
        except KeyboardInterrupt:
            pass

host = "" #お使いのサーバーのホスト名を入れます
port = 11451 #クライアントと同じPORTをしてあげます

#SocketServer.ThreadingTCPServer.allow_reuse_address = True
SocketServer.TCPServer.allow_reuse_address = True
#server = SocketServer.ThreadingTCPServer((host, port), TCPHandler)  
server = SocketServer.TCPServer((host, port), TCPHandler)  
#^Cを押したときにソケットを閉じる
try:
    server.serve_forever()  
except KeyboardInterrupt:
    pass
server.shutdown()
