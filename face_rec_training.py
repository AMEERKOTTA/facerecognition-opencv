import numpy as np
import os
import cv2

import FaceRecognition as fr
print(fr)

# Giving the image path to be recognized #
test_img = cv2.imread(r"H:\Projects\Face Recognition\train-images\1\Alex_Ferguson.jpg")
faces_detected, gray_img = fr.faceDetection(test_img)
print("face Detected: ", faces_detected)

# Training Parts #
faces, faceID = fr.labels_for_training_data(r"H:\Projects\Face Recognition\train-images")
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save(r"H:\Projects\Face Recognition\trainingmodel.yml")

name = {0:"Ameer Shahul Hameed Kotta", 1:"Sir Alex Ferguson"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    label, confidence = face_recognier.predict(roi_gray)
    print("Confidence :", confidence)
    print("Label :", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000,700))

cv2.imshow("face_detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows 
