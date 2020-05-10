import os
import cv2
import time
import platform

from datetime import datetime
from modules.classifier import Classifier

def playsound(file='src/data/alarm.wav'):
    if platform.system().lower() == 'linux':
        os.system("aplay " + file)
    elif 'mac' in platform.system().lower():
        os.system("afplay " + file)
    elif platform.system().lower() == 'windows':
        os.system("fmedia " + file + " --background")

if __name__ == '__main__':
    print("\tWelcome to Super Duper Octo Winner")

    thicc = 2
    score = 0 # To evaluate the state of the driver(drowsy or not)
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        height, width = frame.shape[:2]

        classifier = Classifier(frame)

        left_eye_pred = classifier.left_eye()
        right_eye_pred = classifier.right_eye()

        if left_eye_pred[0] == 0 and right_eye_pred[0] == 0:
            score += 1
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        else:
            score = -1
            cv2.putText(frame, "Opened", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score < 0:
            score = 0
        
        cv2.putText(frame, "Score: "+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        if score > 15: # Using 15 as threshold to say the driver has had his/her eyes closed for too long
            # Driver is feeling sleepy so we play the alarm
            cv2.imwrite(os.path.join(path, str(datetime.now)+'.jpg'), frame)
            playsound() # Play sound

            if thicc < 16:
                thicc += 2
            else:
                thicc -= 2
                if thicc < 2:
                    thicc = 2
            cv2.readOpticalFlow(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
