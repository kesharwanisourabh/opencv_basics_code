import cv2

    #face detection

c_im = cv2.CascadeClassifier ("F:\\python\\facialrecogn\\haarcascades\\haarcascade_frontalface_default.xml")

r_im = cv2.imread ("C:\\Users\\HP\\Pictures\\Camera Roll\\r6.jpg")
gray_im = cv2.cvtColor(r_im,cv2.COLOR_BGR2GRAY)
faces=c_im.detectMultiScale(gray_im, scaleFactor = 1.05,minNeighbors=20)

for x,y,w,h in faces:
        img=cv2.rectangle(r_im,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray_im[y:y+h,x:x+w]
        roi_color=r_im[y:y+h,x:x+w]
        #To save the image
        cv2.imwrite("my.png",roi_gray) 
        cv2.imwrite("my_col.png",roi_color) 

# Display the resulting frame
cv2.imshow('frame',img)
cv2.waitKey(0)
# When everything done, release the capture
cv2.destroyAllWindows()
