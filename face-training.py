import os  # gonna use os.walk, roots, path and files
import cv2
import numpy as n
from PIL import Image  # need to install pillow
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))  # the path of this file
imgs_DIR = os.path.join(DIR, "image")  # the path of the file in which the images are

face_cascade = cv2.CascadeClassifier('Haar_cascade/haar-cascade-files-master/haarcascade_frontalface_alt2.xml')
rec = cv2.face.LBPHFaceRecognizer_create()  # you can use other face recognizer also but i would prefer LBPH as the best
ci = 0  # i means id
label_i = {}
b_label = []
a_training = []

for root, dirs, files in os.walk(imgs_DIR):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):  # defining the type of image to use
            location = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(location)).replace(" ", "-").lower()
            # print(label, location)  # b_label.append(label)  # a_training.append(location)
            if label in label_i:
                pass
            else:
                label_i[label] = ci
                ci += 1
            i = label_i[label]  # here i means the id
            # print(label_i)
            pil_image = Image.open(location).convert('L')  # converting to grayscale
            array_img = n.array(pil_image, "uint8")
            # print(array_img)
            faces = face_cascade.detectMultiScale(array_img, scaleFactor=1.5, minNeighbors=5)

            for (a, b, c, d) in faces:
                roi = array_img[b:b+d, a:a+c]
                a_training.append(roi)
                b_label.append(i)

with open("labels.pickle", "wb") as p:
    pickle.dump(label_i, p)
rec.train(a_training, n.array(b_label))
rec.save("training.xml")

# print(b_label)
# print(a_training)
