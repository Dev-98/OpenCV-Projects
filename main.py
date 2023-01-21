from turtle import color
import cv2

video = cv2.VideoCapture(0)

# The location of haarcascade file 
detector = cv2.CascadeClassifier('C:\\Users\\Dev Gupta\\Documents\\python\\Face recognization\\haarcascade_frontalface_default.xml')


while True:

    ret, frame = video.read()

    if ret:

        faces = detector.detectMultiScale(frame)

        for face in faces:

            x, y, w, h = face
            
            cut = frame[y:y+h , x:x+w]
            fix = cv2.resize(cut,(200,200))
            gray = cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('My screen',frame)
        cv2.imshow('My face',gray)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    # if key == ord('c'):


video.release()
cv2.destroyAllWindows()
