import cv2
import numpy as np 
print("code seen")
#load some pre-trained data on face frontals from opencv('haar cascade algorithm' )
trained_face_data = cv2.CascadeClassifier('C:\\Users\\Farid\\Pictures\\mock up t shirt designs\\haarcascade_frontalface_default.xml')
# choose an image to detect faces in 
#img = cv2.imread('C:\\Users\\Farid\\Pictures\\mock up t shirt designs\\gik.png')  
#img = cv2.imread('C:\\Users\\Farid\\Pictures\\WhatsApp Images\\sent\\twins.jpg')
webcam = cv2.VideoCapture(0)

#iterate forever over frames 
while True:


    ###read current frame 
    successful_frame_read, frame = webcam.read()
    # we must convert to make it greyscale
    #grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangles around the face
    for (x, y, w, h ) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 6)

    cv2.imshow('Fhareed Face Detector',frame)
    key = cv2.waitKey(1)
    #stop if Q key is pressed
    if key==81 or key==113:
        break

## release the video capture object 
webcam.release() 



#Detect faces
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
#print(face_coordinates)
#Draw rectangles around the face
#for (x, y, w, h ) in face_coordinates:
    #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 6)


# Display the image 
#cv2.imshow('Fhareed Face Detector',img)
#cv2.waitKey()
