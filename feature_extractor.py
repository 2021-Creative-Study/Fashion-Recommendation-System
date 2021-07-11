from torchvision import transforms
import torch
import torchvision
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
import glob
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datetime import datetime
import json

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load('./static/test/checkpoint_epoch38181.tar', map_location = torch.device('cpu'))
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        self.model = model.to(self.device)
        model.eval()

    def extract(self, image):
        print(image)
        print('test ...')
        filename = image
        image = Image.open(filename)
        image = image.convert('RGB')
        image = TF.to_tensor(image)
        predict = self.model([image.to(self.device)])
        print('predict값')
        print(predict)
        # mask cpu로 내리기
        pred_mask = predict[0]["masks"][0][0].cpu().detach().numpy()

        # threshold
        threshold = 0.8

        # binary mask로 변환
        pred_mask[pred_mask >= threshold] = 1
        pred_mask[pred_mask < threshold] = 0

        img = cv2.imwrite('img-out.jpg', pred_mask) #이미지 저장

        image_Result = cv2.imread(filename, cv2.IMREAD_COLOR) #덮어쓸 이미지 불러오기

        img = cv2.imread('img-out.jpg',  cv2.IMREAD_GRAYSCALE)
        #ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

        _,contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours 
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        # Draw the contour 
        final = cv2.drawContours(image_Result, contours, contourIdx = -1, 
                                color = (191, 255, 0), thickness = 3)
        filedir2 = 'static/test/' + datetime.now().isoformat().replace(":", ".") + "_"+'2'+'.jpg'
        cv2.imwrite(filedir2,image_Result)

        print(predict)
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        box_num = 0
        min_x = int(predict[0]["boxes"][box_num][0].item())
        min_y = int(predict[0]["boxes"][box_num][1].item())
        max_x = int(predict[0]["boxes"][box_num][2].item())
        max_y = int(predict[0]["boxes"][box_num][3].item())
        print(predict[0]["labels"][0].item())
        cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255, 255), 2)
        # print(datetime.now().isoformat().replace(":", ".") + "_")
        
        print(predict[0]["labels"])
        print(predict[0]["scores"][0].detach())
        
        label = predict[0]["labels"][0].item()
        scores = round(predict[0]["scores"][0].detach().item()*100,2)
        
        filedir = 'static/test/' + datetime.now().isoformat().replace(":", ".") + "_"+'.jpg'
        li = []
        li.append(label)
        li.append(filedir)
        li.append(scores)
        li.append(filedir2)
        cv2.imwrite(filedir,image)
        return li

    def extract2(self, image):
        # print(image)
        filename = image
        image = Image.open(filename)
        image = image.convert('RGB')
        image = TF.to_tensor(image)
        predict = self.model([image.to(self.device)])
        # print(predict[0]["labels"][0].item())
        # print(predict[0]["labels"])
        label = predict[0]["labels"][0].item()
        return label

    
    
        
        








