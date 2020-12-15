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

def handDetection(frame,originale):
    kernel_blur=7
    seuil=15    
    surface=5000
    kernel_dilate=np.ones((5, 5), np.uint8)
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
    iw, ih, inference_time, results = yolo.inference(frame)
    for detection in results: 
        id, name, confidence, x, y, w, h = detection 
        cx=x+(w/2)      

        #Adding a padding to get the whole hand
        centerX = int((2*x+w)/2)
        centerY = int((2*y+h)/2)
        H=int(1.5*h)
        W=int(1.5*w)
        X=max(0,int(centerX-W/2))
        Y=max(int(centerY-H/2),0)
        
        

    return  X,Y,H,W     

