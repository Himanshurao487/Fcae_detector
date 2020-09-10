import numpy as n
import cv2
import pickle

# importing the opencv and numpy above and getting cascade file below
face_cascade = cv2.CascadeClassifier('Haar_cascade/haar-cascade-files-master/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('Haar_cascade/haar-cascade-files-master/haarcascade_eye_tree_eyeglasses.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("training.xml")

label = {"name": 1}  # here l is the label
with open("labels.pickle", "rb") as p:
    b_labels = pickle.load(p)
    label = {h: r for r, h in b_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()  # the frame to capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    green = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (a, b, c, d) in faces:  # here a and b are the two lengths of rectangle and c is width and d is height
        print(a, b, c, d)
        roi_gray = gray[b:b+d, a:a+c]  # let say animal as the variables (animal1-run, animal2-finish)
        roi_colorful = frame[b:b+d, a:a+c]  # let say animal as the variables (animal1-run, animal2-finish)
        __id__, cof = rec.predict(roi_gray)
        if cof >= 45:  # and cof <= 85
            print(__id__)
            print(label[__id__])
            text_font = cv2.FONT_HERSHEY_DUPLEX  # You can take any sort of font as per your reference
            text_name = label[__id__]
            text_color = (255, 0 , 0)  # You can take any sort of color as per your reference
            text_thickness = 1
            cv2.putText(frame, text_name, (a, b), text_font, 1, text_color, text_thickness, cv2.LINE_AA)
        img_item = "Himanshu.png"  # saving img path with only face
        cv2.imwrite(img_item, roi_colorful)  # take this roi as the color of the picture of yours in the image folder

        diagram_color = (65535, 65280, 16777215)  # color for the face detector frame
        thickness_line = 3  # you can add a thickness of the border as you want
        animal_1 = a + c
        animal_2 = b + d
        cv2.rectangle(frame, (a, b), (animal_1, animal_2), diagram_color, thickness_line)
        # if you want a circle you can do some changes on face capture diagram as you like

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything is done, then release the capture
cap.release()
cv2.destroyAllWindows()