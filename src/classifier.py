import cv2
from time import sleep

class Classifier(object):
    def __init__(self, img):
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
            #cv2.rectangle(right_eye, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # TODO: Perform data augmentation and fit to model and return prediction

    def left_eye(self):
        for x, y, w, h in self.left_eye:
            left_eye_img = self.img[y:y+h, x:x+w]
            cv2.rectangle(left_eye_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.rectangle(right_eye, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # TODO: Perform data augmentation and fit to model and return prediction
        

        label = ['close', 'open']
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = left_eye_cascade.detectMultiScale(gray)
        right_eye = right_eye_cascade.detectMultiScale(gray)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

