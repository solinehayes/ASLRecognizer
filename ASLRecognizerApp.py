#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from VideoCapture import VideoCapture
import tkinter
import PIL.Image, PIL.ImageTk
import torchvision.transforms as tt

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']



def fromImageToTensor(image):
    model_transform = tt.Compose([
                             tt.ToTensor(),
                             tt.Resize((224,224)),
                             tt.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229]),
                         ])
    return model_transform(image)

def getModelPrediction(model, image):
    imageTensor = fromImageToTensor(image)
    imageTensor = imageTensor.reshape(1,3,224,224)
    response = model(imageTensor)
    _, predc = response.topk(1, 1, True, True)
    
    return classes[int(predc.numpy()[0])]

class ASLRecognizerApp:
    def __init__(self, window, window_title, model, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.model = model
        
        self.video_source = video_source
        
         # Adding button to snap (to be taken off eventually)
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.gestureDetection)
        self.btn_snapshot.pack(anchor=tkinter.N, expand=True)
        
        self.setupMiddleFrame()
        
        # create message display
        self.message=""
        self.textDisplay = tkinter.Text(window)
        self.textDisplay.pack(anchor= tkinter.S, expand=True)
        
        self.delay = 2
        self.update()
        
        self.window.mainloop()

    def setupMiddleFrame(self):
        self.middleFrame = tkinter.Frame(self.window)
        
        #Setting up the video capture
        self.vid = VideoCapture(video_source)
        self.canvas = tkinter.Canvas(self.middleFrame, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(side=tkinter.LEFT)
        
        #Setting up the documentation
        alphabet = PIL.Image.open("./assets/ASLAlphabet.jpg")
        imwidth, imheight = alphabet.size
        ratio = imwidth/imheight
        alphabet = alphabet.resize((int(ratio*700), 700), PIL.Image.ANTIALIAS)
        image = PIL.ImageTk.PhotoImage(alphabet)
        self.labelAlphabet= tkinter.Label(self.middleFrame,image=image)
        self.labelAlphabet.pack(side=tkinter.LEFT)
        
        self.middleFrame.pack()
        
    def setMessageDisplay(self):
        self.textDisplay.delete(1.0,"end")
        self.textDisplay.insert(1.0, self.message)
        
    def update(self):
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0, anchor = tkinter.NW, image=self.photo)
        self.window.after(self.delay, self.update)

    def gestureDetection(self):
        ret, frame = self.vid.get_frame()
        if ret:
            letter= getModelPrediction(self.model,frame)
            if(letter == "space"):
                self.message +=" "
            elif (letter == "del"):
                if(len(self.message)>0):
                    self.message = self.message[0:len(self.message)-1]
            elif (len(letter)==1):
                self.message+=letter 
            self.setMessageDisplay()
            