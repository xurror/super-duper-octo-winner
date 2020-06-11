import os
import cv2
import time
import platform

from datetime import datetime
from classifier import Classifier
from models.VGG_Face import Vgg_face_dag
from models.VGG_Face import vgg_face_dag

def playsound(file='src/data/alarm.wav'):
    # TODO; Libraries like playsound( Use a better sound method)
    if platform.system().lower() == 'linux':
        os.system("aplay " + file)
    elif 'mac' in platform.system().lower():
        os.system("afplay " + file)
    elif platform.system().lower() == 'windows':
        os.system("fmedia " + file + " --background")

# TODO: Set a thicker frame around the face with a bighter color

def main():
    VGG_Face = Vgg_face_dag()
    VGG_Face = vgg_face_dag(VGG_Face, "src/models/vgg_face_dag.pth")
    thicc = 2
    score = 0 # To evaluate the state of the driver(drowsy or not)
    frame_count = 0
    frames = []
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        classifier = Classifier(frame)

        left_eye_pred = classifier.left_eye()
        right_eye_pred = classifier.right_eye()

        frames.append(frame)
        if len(frames) == 10:
            drunk_pred = classifier.drunk_pred(frames, VGG_Face)
            frames = []

        if drunk_pred == 1:
            cv2.putText(frame, "Drunk", (width-10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Sober", (width-10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if left_eye_pred == 0 and right_eye_pred == 0:
            score += 1
            cv2.putText(frame, "Asleep", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        else:
            score = -1
            cv2.putText(frame, "Awake", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        
        cv2.putText(frame, "Score: "+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        if score >= 8: # Using 8 as threshold to say the driver has had his/her eyes closed for too long
            # Driver is feeling sleepy so we play the alarm
            #cv2.imwrite(os.path.join(path, str(datetime.now)+'.jpg'), frame) 
            # TODO: Fix code crash after sound play
            playsound() # Play sound

            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.readOpticalFlow(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27:
    	    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("\tWelcome to Super Duper Octo Winner")
    main()
