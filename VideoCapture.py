#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from hand_detection import demo

class VideoCapture:

    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        originale = self.vid.read()
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    def get_frame(self,num):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            X,Y,H,W=demo(frame,originale)   
            originale=frame
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),X,Y,H,W)
            else:
                return (ret, None,X,Y,H,W)
        else:
            return (ret, None,X,Y,H,W)
