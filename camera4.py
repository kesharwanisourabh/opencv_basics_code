#for video with face detection 
import numpy as np
import cv2

c_im = cv2.CascadeClassifier ("F:\\python\\facialrecogn\\haarcascades\\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    r_img=frame 
    # Our operations on the frame come here
    gray_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=c_im.detectMultiScale(gray_im, scaleFactor = 1.05,minNeighbors=20)
    for x,y,w,h in faces:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        r_img=img 
    # Display the resulting frame
    cv2.imshow('Demo',r_img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
