#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import tkinter
from ASLRecognizerApp import ASLRecognizerApp


#Import model
model = torchvision.models.resnet34(pretrained = True)
model.fc = torch.nn.Linear(model.fc.in_features,29)
model.load_state_dict(torch.load("./CNNModel.pth"))
model.eval()

ASLRecognizerApp(tkinter.Tk(), "ASLRecognizer", model)
print("test")
