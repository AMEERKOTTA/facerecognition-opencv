import numpy as np
import os
import cv2

import FaceRecognition as fr
print(fr)

# Give path to the image to be tested #
test_img = cv2.imread(r"H:\Projects\Face Recognition\train-images\1\Alex_Ferguson.jpg")

faces_detected, gray_img = fr.faceDetection(test_img)
print("face Detected:", faces_detected)

# Give path to the training model #
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"H:\Projects\Face Recognition\trainingmodel.yml")

# mention the categories of images to be recognized #
name = {0:"Ameer Shahul Hameed Kotta", 1:"Sir Alex Ferguson"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("Confidence:", confidence)
    print("Label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

# Resize the Image #
resized_img = cv2.resize(test_img, (1000,700))

cv2.imshow("face detection:", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows 
