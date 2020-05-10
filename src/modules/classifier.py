import cv2
import numpy as np

from tensorflow.keras.models import load_model

class Classifier(object):
    def __init__(self, img):
        self.model = load_model('src/models/cnnCat2.h5')

        self.img = img
        self.face_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_frontalface_default.xml')
        self.left_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier('src/haarcascades/haarcascade_righteye_2splits.xml')

        self.height, self.width = self.img.shape[:2] 

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.faces = self.face_cascade.detectMultiScale(self.gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        self.left_eye = self.left_eye_cascade.detectMultiScale(self.gray)
        self.right_eye =  self.right_eye_cascade.detectMultiScale(self.gray)

        cv2.rectangle(self.img, (0, self.height-50) , (200, self.height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(self.img, (x, y) , (x+w, y+h) , (100, 100, 100), 1 )

    def right_eye(self):
        for x, y, w, h in self.right_eye:
            right_eye_img = self.img[y:y+h, x:x+w]

            right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
            right_eye_img = cv2.resize(right_eye_img, (24, 24)) # resize the image to 24*24 pixels
            right_eye_img = right_eye_img / 255 # normalize our data for better convergence
            right_eye_img = right_eye_img.reshape(24, 24, -1)
            right_eye_img = np.expand_dims(right_eye_img, axis=0)

            return self.model.predict_classes(right_eye_img)

    def left_eye(self):
        for x, y, w, h in self.left_eye:
            left_eye_img = self.img[y:y+h, x:x+w]
            cv2.rectangle(left_eye_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2GRAY)
            left_eye_img = cv2.resize(left_eye_img, (24, 24))
            left_eye_img = left_eye_img / 255
            left_eye_img = left_eye_img.reshape(24, 24, -1)
            left_eye_img = np.expand_dims(left_eye_img, axis=0)

            return self.model.predict_classes(left_eye_img)
