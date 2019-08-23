#To detect the faces of mobile cam
import numpy as np
import cv2
import urllib.request

c_im = cv2.CascadeClassifier ("F:\\python\\facialrecogn\\haarcascades\\haarcascade_frontalface_default.xml")

while(True):
    cap = urllib.request.urlopen("http://192.168.1.5:8080/shot.jpg")
    imnp=np.array(bytearray(cap.read()),dtype=np.uint8)
    frame=cv2.imdecode(imnp,-1)
    r_img=frame 
    # Our operations on the frame come here
    gray_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=c_im.detectMultiScale(gray_im, scaleFactor = 1.05,minNeighbors=20)
    for x,y,w,h in faces:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        r_img=img 
        roi_gray=gray_im[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #To save the image
        cv2.imwrite("my.png",roi_gray) 
        cv2.imwrite("my_col.png",roi_color) 
    # Display the resulting frame
    cv2.imshow('Demo',r_img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    

cv2.destroyAllWindows()
    
