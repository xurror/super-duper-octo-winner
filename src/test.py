import os
import cv2
from datetime import datetime
from classifier import Classifier

def test_with_image(img_path):
    thicc = 2
    score = 0 # To evaluate the state of the driver(drowsy or not)
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    img = cv2.imread(img_path)

    height, width = img.shape[:2]

    classifier = Classifier(img)

    left_eye_pred = classifier.left_eye()

    right_eye_pred = classifier.right_eye()

    if left_eye_pred == 0 and right_eye_pred == 0:
        score += 1
        cv2.putText(img, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        score = -1
        cv2.putText(img, "Opened", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(img, "Score: "+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15: # Using 15 as threshold to say the driver has had his/her eyes closed for too long
        # Driver is feeling sleepy so we play the alarm
        cv2.imwrite(os.path.join(path, str(datetime.now)+'.jpg'), img)
        #playsound() # Play sound

        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.readOpticalFlow(img, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

img_path = input("Enter image path: ")
test_with_image(img_path)