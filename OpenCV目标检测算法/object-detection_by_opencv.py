import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import math




cap = cv2.VideoCapture('./video1.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()# 创建高斯模型混合对象
thresh = 200

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
while True:
    ret,frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame) # 获取前景的蒙版
    # cv2.imshow('mask',fgmask)
    # cv2.waitKey(300)
    # cv2.destroyAllWindows()
    _,fgmask = cv2.threshold(fgmask,30,0xff,cv2.THRESH_OTSU) #对前景蒙版进行二值化处理，前景为白色
    bgImage = fgbg.getBackgroundImage() #获取背景
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    # cv2.imshow('mask',fgmask)
    # cv2.waitKey(300)
    # cv2.destroyAllWindows()


    cnts,_ = cv2.findContours(fgmask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #外轮廓检测，把白色的轮廓框出来，即相当于得到了运动的物体，最关键的部分在上面，Opencv牛逼，直接给一个函数帮你做了最难的事情
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if (area<thresh):
            continue
        count += 1
        
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0xff,0),2)
    print('共检测到：',count,'个目标','\n')
    cv2.imshow('frame',frame)
    cv2.imshow('Background',bgImage)
    
    key = cv2.waitKey(30)
    if key==27:
        break
        