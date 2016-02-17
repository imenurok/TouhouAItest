# -*- coding:utf-8 -*-
import socket
import ctypes
from PIL import ImageGrab
import cv2
import numpy
import time
import random

#DirectXにコマンド送る関数
SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ScreenSchotter():
    x=284
    y=290
    w=670
    h=740
    img = ImageGrab.grab((x,y,w,h))
    img = numpy.asarray(img,dtype='uint8')
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #jpegstring=cv2.cv.EncodeImage('.jpeg',cv2.cv.fromarray(img)).tostring() 
    jpegstring=img.tostring()
    client.send(str(len(jpegstring)))
    client.recv(1024)
    client.send(jpegstring) #適当なデータを送信します（届く側にわかるように）
    response = client.recv(1024).strip() #レシーブは適当な2進数にします（大きすぎるとダメ）
    return int(response)

def commandStart(response):
    if response==-1:
        PressKey(0x01)#ESC
    elif response==0:
        PressKey(0x2c)#Z
    elif response==1:
        PressKey(0x2c)#Z
        PressKey(0xcb)#LEFT
    elif response==2:
        PressKey(0x2c)#Z
        PressKey(0xc8)#UP
    elif response==3:
        PressKey(0x2c)#Z
        PressKey(0xcd)#RIGHT
    elif response==4:
        PressKey(0x2c)#Z
        PressKey(0xd0)#DOWN
    elif response==5:
        PressKey(0x2c)#Z
        PressKey(0xcb)#LEFT
        PressKey(0xc8)#UP
    elif response==6:
        PressKey(0x2c)#Z
        PressKey(0xc8)#UP
        PressKey(0xcd)#RIGHT
    elif response==7:
        PressKey(0x2c)#Z
        PressKey(0xcd)#RIGHT
        PressKey(0xd0)#DOWN
    elif response==8:
        PressKey(0x2c)#Z
        PressKey(0xd0)#DOWN
        PressKey(0xcb)#LEFT
    elif response==9:
        #LSHIFT
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
    elif response==10:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xcb)#LEFT
    elif response==11:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xc8)#UP
    elif response==12:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xcd)#RIGHT
    elif response==13:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xd0)#DOWN
    elif response==14:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xcb)#LEFT
        PressKey(0xc8)#UP
    elif response==15:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xc8)#UP
        PressKey(0xcd)#RIGHT
    elif response==16:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xcd)#RIGHT
        PressKey(0xd0)#DOWN
    elif response==17:
        PressKey(0x2a)#LSHIFT
        PressKey(0x2c)#Z
        PressKey(0xd0)#DOWN
        PressKey(0xcb)#LEFT

def commandEnd(response):
    if response==-1:
        ReleaseKey(0x01)#ESC
    elif response==0:
        ReleaseKey(0x2c)#Z
    elif response==1:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcb)#LEFT
    elif response==2:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xc8)#UP
    elif response==3:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcd)#RIGHT
    elif response==4:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xd0)#DOWN
    elif response==5:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcb)#LEFT
        ReleaseKey(0xc8)#UP
    elif response==6:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xc8)#UP
        ReleaseKey(0xcd)#RIGHT
    elif response==7:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcd)#RIGHT
        ReleaseKey(0xd0)#DOWN
    elif response==8:
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xd0)#DOWN
        ReleaseKey(0xcb)#LEFT
    elif response==9:
        #LSHIFT
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
    elif response==10:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcb)#LEFT
    elif response==11:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xc8)#UP
    elif response==12:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcd)#RIGHT
    elif response==13:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xd0)#DOWN
    elif response==14:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcb)#LEFT
        ReleaseKey(0xc8)#UP
    elif response==15:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xc8)#UP
        ReleaseKey(0xcd)#RIGHT
    elif response==16:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xcd)#RIGHT
        ReleaseKey(0xd0)#DOWN
    elif response==17:
        ReleaseKey(0x2a)#LSHIFT
        ReleaseKey(0x2c)#Z
        ReleaseKey(0xd0)#DOWN
        ReleaseKey(0xcb)#LEFT

host = "" #お使いのサーバーのホスト名を入れます
port = 11451 #適当なPORTを指定してあげます
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port)) #これでサーバーに接続します

try:
    response=0
    time.sleep(4)
    commandStart(-1)
    time.sleep(4*(1.0/60))
    commandEnd(-1)
    while True:
        for no in range(10):
            time.sleep(1)
            while True:
                commandStart(response)
                new_response=ScreenSchotter()
                commandEnd(response)
                if new_response==-2:
                    print "end"
                    break
                else:
                    response=new_response
            client.send("stop")
            client.recv(1024)
            time.sleep(6)
            response=-1
            commandStart(response)
            time.sleep(4*(1.0/60))
            commandEnd(response)
            response=random.randint(0,17)
except KeyboardInterrupt:
    client.close()