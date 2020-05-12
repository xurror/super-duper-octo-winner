import cv2
import torch
import numpy as np

import torchvision.transforms as transforms
from model import Net
from PIL import Image

def extract_eye():
    # TODO: Manually perform feature extraction(optional)
    pass

class Classifier(object):
    def __init__(self, img):
        self.model = Net()
        self.model.load_state_dict(torch.load('src/models/model.pt'))
        self.transform = transforms.Compose([
            transforms.Resize(size=(24,24)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.use_cuda = torch.cuda.is_available()

        self.img = img
        face_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_frontalface_default.xml')
        left_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_lefteye_2splits.xml')
        right_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_righteye_2splits.xml')

        height, width = self.img.shape[:2] 

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        self.left_eye_gray = left_eye_cascade.detectMultiScale(gray)
        self.right_eye_gray =  right_eye_cascade.detectMultiScale(gray)
        
        cv2.rectangle(self.img, (0, height-50) , (200, height) , (0,0,0) , thickness=cv2.FILLED )
        
        for (x,y,w,h) in faces:
            cv2.rectangle(self.img, (x, y) , (x+w, y+h) , (100, 100, 100), 1)

    def right_eye(self):
        for x, y, w, h in self.right_eye_gray:
            right_eye_img = self.img[y:y+h, x:x+w]
            
            right_eye_img = Image.fromarray(right_eye_img)

            right_eye_img = self.transform(right_eye_img)
            right_eye_img = torch.unsqueeze(right_eye_img, 0)

            output = self.model(right_eye_img)
            _, pred = torch.max(output, 1)
            pred_index = np.squeeze(pred.numpy()) if not self.use_cuda else np.squeeze(pred.cpu().numpy())
            
            return pred_index

    def left_eye(self):
        for x, y, w, h in self.left_eye_gray:
            left_eye_img = self.img[y:y+h, x:x+w]

            left_eye_img = Image.fromarray(left_eye_img)

            left_eye_img = self.transform(left_eye_img)

            left_eye_img = torch.unsqueeze(left_eye_img, 0)

            output = self.model(left_eye_img)
            _, pred = torch.max(output, 1)
            pred_index = np.squeeze(pred.numpy()) if not self.use_cuda else np.squeeze(pred.cpu().numpy())
            
            return pred_index
