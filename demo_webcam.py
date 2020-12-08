import argparse
import cv2
import os
import sys
import time
import numpy as np
from PIL import Image
from yolo import YOLO
from skimage.measure import compare_ssim
from skimage.transform import resize
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn / v4-tiny')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

def rsz(image):
    basewidth = 200
    img = image
    wpercent = (basewidth/float(img.shape[0]))
    hsize = int((float(img.shape[1])*float(wpercent)))
    img = resize(img, (basewidth,hsize),anti_aliasing=True)

#    img = npimg.resize(, Image.ANTIALIAS,refcheck=False)
    return img


vidcap=cv2.VideoCapture(0)
kernel_blur=7
seuil=10
surface=5000
ret, originale=vidcap.read()
originale=cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
originale=cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
original=cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
kernel_dilate=np.ones((7, 7), np.uint8)
s=50
def fill_check(x,y,x1,y1,s):
    Z=np.zeros((s,s,2))
    t=0
    for i in range(s):
        Z[i,:,0]=np.arang(x-2,x+s-2)
        Z[i,:,1]=np.arang(y-2,y+s-2)
    for i in range(s):
        for j in range(s):
            if(Z[i,j,:]==(x1,y1)):
                t=1        
    return t            

while True:
    ret, frame=vidcap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)
    mask=cv2.absdiff(originale, gray)
    mask=cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    mask=cv2.dilate(mask, kernel_dilate, iterations=3)
    contours, nada=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contour=frame.copy()
    T=[]
    for c in contours:

        if cv2.contourArea(c)<surface:
            continue
        T.append(c)
    print("len(cnt) = ",len(T))    
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
    if (d==1):            
        cv2.rectangle(frame, (min(x1,x), min(y1,y)), (min(x1,x)+int(1.25*max(w1,w)), min(y1,y)+int(1.25*max(h1,h))), (0, 0, 255), 2)
        text = "(%s)" % round(confidence, 2)
        cv2.putText(frame, text, (min(x1,x), min(y1,y) - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
        cv2.putText(frame, "[o|l]seuil: {:d}  [p|m]blur: {:d}  [i|k]surface: {:d}".format(seuil, kernel_blur, surface), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        cv2.imshow("frame", frame)

            #cv2.imshow("Im1",Im1)
            #cv2.imshow("Im2",Im2)
#            L.append(Im1)
#            L.append(Im2)
#            rectList, weights=cv2.groupRectangles(L,1, 0.2)
#            print(len(rectList))
#            Im2=resize(Im2, (Im1.shape[0],Im1.shape[1],3),anti_aliasing=True)
#            print(Im1.shape,Im2.shape)
#            L= cv2.bitwise_and(Im1, Im2)
#            cv2.putText(L, "[o|l]seuil: {:d}  [p|m]blur: {:d}  [i|k]surface: {:d}".format(seuil, kernel_blur, surface), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
#           cv2.imshow("Im1",L)   
#                cv2.imshow("Im22", Im2)
#                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
#                cv2.imshow("frame", frame)
 
#            print(Im2.shape,Im1.shape)

#            print(Im2.shape)
    '''
            grayA = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)
            (score, diff) = compare_ssim(grayA,grayB, full=True)           
            if (score>=0.5):
                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x1+int(w1), y1+int(h1)), color, 2)
                text = "(%s)" % round(confidence, 2)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                cv2.putText(frame, "[o|l]seuil: {:d}  [p|m]blur: {:d}  [i|k]surface: {:d}".format(seuil, kernel_blur, surface), (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
                cv2.imshow("frame", frame)
    '''    
    originale=gray
    key=cv2.waitKey(10)&0xFF
    intrus=0
    if key==ord('q'):
        break
    if key==ord('p'):
        s+=5
        print("s",s)
    if key==ord('m'):
        s-=5
        print("s",s)        
    if key==ord('i'):
        surface+=1000
        print("surface",surface)        
    if key==ord('k'):
        surface-=500
        print("surface",surface)        
    if key==ord('o'):
        seuil=min(255, seuil+1)
    if key==ord('l'):
        seuil=max(1, seuil-1)    
    
vidcap.release()
cv2.destroyAllWindows()        
