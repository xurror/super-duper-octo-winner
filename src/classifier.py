import cv2
import torch
import numpy as np

import torchvision.transforms as transforms
from models.sleep_model import SleepNet
from models.drunk_model import DrunkNet
from models.VGG_Face import get_features_vector
from PIL import Image

class Classifier(object):
    def __init__(self, img):
        self.sleep_model = SleepNet()
        self.sleep_model.load_state_dict(torch.load('src/models/sleep_model.pt'))
        self.drunk_model = DrunkNet()
        self.drunk_model.load_state_dict(torch.load('src/models/drunk_model.pt'))
        self.transform = transforms.Compose([
            transforms.Resize(size=(24,24)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.use_cuda = torch.cuda.is_available()

        self.img = img
        self.face_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_frontalface_default.xml')
        left_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_lefteye_2splits.xml')
        right_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_righteye_2splits.xml')
        eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_eye.xml')

        height, width = self.img.shape[:2] 

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(self.gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        self.left_eye_gray = left_eye_cascade.detectMultiScale(self.gray)
        self.right_eye_gray =  right_eye_cascade.detectMultiScale(self.gray)
        
        cv2.rectangle(self.img, (0, height-50) , (200, height) , (0,0,0) , thickness=cv2.FILLED )
        
        for (x,y,w,h) in faces:
            cv2.rectangle(self.img, (x, y) , (x+w, y+h) , (255, 0, 0), 2)
            roi_gray = self.gray[y:y+h, x:x+w]
            roi_color = self.img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (x,y,w,h) in eyes:
                cv2.rectangle(roi_color, (x, y), (x+w, y+h),(0,255,0), 2)

    def drunk_pred(self, frames, net):
        preds = []
        for frame in frames:
            faces = self.face_cascade.detectMultiScale(frame, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
            for (x,y,w,h) in faces:
                face = frame[y:y+h, x:x+w]
                face = Image.fromarray(face)
                face_vec = get_features_vector(net, face)
                output = self.drunk_model(face_vec)
                _, pred = torch.max(output, 1)
                pred_index = np.squeeze(pred.numpy()) if not self.use_cuda else np.squeeze(pred.cpu().numpy())
                preds.append(pred_index)
        try:
            return round(sum(preds) / len(preds))
        except ZeroDivisionError:
            return 0

    def right_eye(self):
        for x, y, w, h in self.right_eye_gray:
            right_eye_img = self.img[y:y+h, x:x+w]
            right_eye_img = Image.fromarray(right_eye_img)

            right_eye_img = self.transform(right_eye_img)
            right_eye_img = torch.unsqueeze(right_eye_img, 0)

            output = self.sleep_model(right_eye_img)
            _, pred = torch.max(output, 1)
            pred_index = np.squeeze(pred.numpy()) if not self.use_cuda else np.squeeze(pred.cpu().numpy())
            
            return pred_index

    def left_eye(self):
        for x, y, w, h in self.left_eye_gray:
            left_eye_img = self.img[y:y+h, x:x+w]
            left_eye_img = Image.fromarray(left_eye_img)

            left_eye_img = self.transform(left_eye_img)

            left_eye_img = torch.unsqueeze(left_eye_img, 0)

            output = self.sleep_model(left_eye_img)
            _, pred = torch.max(output, 1)
            pred_index = np.squeeze(pred.numpy()) if not self.use_cuda else np.squeeze(pred.cpu().numpy())
            
            return pred_index
