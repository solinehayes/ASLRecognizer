#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from hand_detection import handDetection

class VideoCapture:

    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        ret, self.originale = self.vid.read()
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_iter =0 
        self.max_frame_iter=10000
        self.boundingBox = [0,0,self.height,self.width]


    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    def get_frame(self):
        self.frame_iter+=1
        if(self.frame_iter>self.max_frame_iter):
            self.frame_iter=0
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if(self.frame_iter%5==0):
                X,Y,H,W=handDetection(frame,self.originale) 
                self.boundingBox =[X,Y,H,W]
            self.originale=frame
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),self.boundingBox)
            else:
                return (ret, None,self.boundingBox)
        else:
            return (ret, None,self.boundingBox)
