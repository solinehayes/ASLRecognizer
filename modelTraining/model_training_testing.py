#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as tt
import torchvision
import training_functions

train_transform = tt.Compose([tt.RandomCrop(200, padding=25, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(10),
                         tt.RandomPerspective(distortion_scale=0.2),
                         tt.ToTensor(),
                         tt.Resize((224,224)),
                         tt.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229]),
                         ])
test_transform = tt.Compose([
                             tt.ToTensor(),
                             tt.Resize((224,224)),
                             tt.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229]),
                         ])

#Former path in the google colab
PATH = "/content/drive/My Drive/ComputerVision/" 

#Creating model
model = torchvision.models.resnet34(pretrained = True)
model.fc = torch.nn.Linear(model.fc.in_features,29)

#Load data
dataset = torchvision.datasets.ImageFolder(root=PATH+"tmp/asl_alphabet_train/asl_alphabet_train/",transform = train_transform)
dataset_test = torchvision.datasets.ImageFolder(root=PATH+"tmp/asl_alphabet_test/asl_alphabet_test/",transform = test_transform)

test_data = torchvision.datasets.ImageFolder(root=PATH+"tmp_test/", transform = test_transform)

#Prepare data
train_data, valid_data = training_functions.prepareDatasets(10000,500,dataset)
train_loader, valid_loader = training_functions.createDataLoaders(train_data, valid_data, 100)

#Prepare training
model = model.to(training_functions.get_default_device())
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")


#Training
epoch = 0

for i in range(5):
    epoch += 1
    training_functions.train(model, train_loader, optimizer, criterion, epoch, training_functions.get_default_device(), logging_freq=1)
    acc = training_functions.validation(model, valid_loader, criterion, training_functions.get_default_device())
    

#Testing the model
real_test_data = torchvision.datasets.ImageFolder(root=PATH+"realTest/", transform = test_transform)

training_functions.displayTestInformation(model,test_data,True)
training_functions.displayTestInformation(model,real_test_data,True)

#Saving the model
model = model.to("cpu")
torch.save(model.state_dict(), PATH+"models/pretrainedModelFullMixed-Raph-dict.pth")

    