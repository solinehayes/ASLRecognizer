import cv2
import os
import sys
import time
import numpy as np
from PIL import Image
from yolo import YOLO
from skimage.measure import compare_ssim
from skimage.transform import resize


yolo = YOLO("YOlOModels/cross-hands-tiny-prn.cfg", "YOLOModels/cross-hands-tiny-prn.weights", ["hand"])

yolo.size = 416
yolo.confidence = 0.2

def demo(frame,originale):
    kernel_blur=7
    seuil=15    
    surface=5000
    kernel_dilate=np.ones((7, 7), np.uint8)
    s=50
    originale=cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
    originale=cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)
    mask=cv2.absdiff(originale, gray)
    mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    mask=cv2.dilate(mask, kernel_dilate, iterations=3)
    contours, nada=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contour=frame.copy()
    T=[]
    X,Y,H,W =0,0,0,0
    for c in contours:
        if cv2.contourArea(c)<surface:
            continue
        T.append(c)
    iw, ih, inference_time, results = yolo.inference(frame)
    d=0
    for detection in results: 
        id, name, confidence, x, y, w, h = detection 
        cx=x+(w/2)        
        for cnt in T:
            x1, y1, w1, h1=cv2.boundingRect(cnt)
            cx1=x1+(w1/2)
            if(abs(cx1-cx)<=s):
                d=1
                break
        if d==1:            
            X=min(x1,x)
            Y=min(y1,y)
            H=int(1.25*max(h1,h))
            W=int(1.25*max(w1,w))
        

    return  X,Y,H,W     

