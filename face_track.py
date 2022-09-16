import cv2
import dlib
import numpy as np
import keyboard


video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\Dev Gupta\\Documents\\python\\Face recognization\\shape_predictor_68_face_landmarks.dat')

while True:

    ret,frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces = detector(gray)
    
    for face in faces:

        landmarks = predictor(gray,face)
        ori = landmarks.parts()[27]

        lip_up = landmarks.parts()[62].y
        lip_d =  landmarks.parts()[66].y

        if lip_d - lip_up > 3:
            cv2.putText(frame,'Open',(ori.x,ori.y),cv2.FONT_HERSHEY_DUPLEX,3,(255,255,0),3)
            keyboard.press('up')
        else :
            cv2.putText(frame,'Close',(ori.x,ori.y),cv2.FONT_HERSHEY_DUPLEX,3,(255,255,0),3)
            keyboard.press('down')

        for points in landmarks.parts():

            cv2.circle(frame,(points.x, points.y),2,(255,255,0),2)


    if ret :

            
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)

        cv2.imshow(' ',frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()