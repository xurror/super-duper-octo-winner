import os
import cv2
from datetime import datetime
from classifier import Classifier
from models.VGG_Face import Vgg_face_dag
from models.VGG_Face import vgg_face_dag

def test_with_image(img_path):
    VGG_Face = Vgg_face_dag()
    VGG_Face = vgg_face_dag(VGG_Face, "src/models/vgg_face_dag.pth")
    thicc = 2
    score = 0 # To evaluate the state of the driver(drowsy or not)
    frame_count = 0
    frames = []
    path = os.getcwd()
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    img = cv2.imread(img_path)

    height, width = img.shape[:2]

    classifier = Classifier(img)

    left_eye_pred = classifier.left_eye()

    right_eye_pred = classifier.right_eye()

    frames.append(img)
    drunk_pred = classifier.drunk_pred(frames, VGG_Face)

    if drunk_pred == 1:
        cv2.putText(img, "Drunk", (10, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "Sober", (10, 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
    if left_eye_pred == 0 and right_eye_pred == 0:
        score += 1
        cv2.putText(img, "Asleep", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        score = -1
        cv2.putText(img, "Awake", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(img, "Score: "+str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 8: # Using 15 as threshold to say the driver has had his/her eyes closed for too long
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
    k = cv2.waitKey(0)
    if k == 27:
    	cv2.destroyAllWindows()

img_path = input("Enter image path: ")
test_with_image(img_path)
