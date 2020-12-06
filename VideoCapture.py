#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

class VideoCapture:
    def __init__(self, video_source=0, video_height = 700, video_width=700):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = video_width
        self.height = video_height

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame[0:self.width,0:self.height], cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
