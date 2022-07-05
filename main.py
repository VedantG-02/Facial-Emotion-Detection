# import tensorflow as tf
import keras
import cv2
import numpy as np

# loading the model
model = keras.models.model_from_json(open("my_model1.json", "r").read())
model.load_weights('my_model1.h5')

# to access any video
# cap = cv2.VideoCapture('') # insert path or video name inside quotes 
# to access webcam
cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotions = ('angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral')

########################################### FOR VIDEOS #####################################################

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 6)
    try:
        for (x, y, w, h) in faces:
            img = gray[x:x+w, y:y+h]
            # converting into (48, 48) sized image
            img = cv2.resize(img, (48, 48))
            # predicting emotion on detected face
            img = img/255
            prediction = model.predict(img.reshape(1, 48, 48))
            idx = np.argmax(prediction)

            # angry - red
            if(idx==0): 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA) 
            # disgusted - yellow
            elif(idx==1):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # fearful - blue
            elif(idx==2):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # happy - green
            elif(idx==3):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # neutral - black
            elif(idx==4):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # sad - orange
            elif(idx==5):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 255, 255), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # surprised - cyan
            elif(idx==6):
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                frame = cv2.putText(frame, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        pass
        
    cv2.imshow('frame', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

cv2.destroyAllWindows()

########################################### FOR IMAGES #####################################################

'''
img = cv2.imread('', 1) # Insert path or image name inside quotes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    # print(x, y, w, h)
    if x+w <= gray.shape[0] and y+h <= gray.shape[1]:
        img1 = gray[x:x+w, y:y+h]
        # converting into (48, 48) sized image
        img1 = cv2.resize(img1, (48, 48))
        # predicting emotion on detected face
        img1 = img1/255
        prediction = model.predict(img1.reshape(1, 48, 48))
        idx = np.argmax(prediction)
        # print(emotions[idx])

        # angry - red
        if(idx==0): 
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA) 

        # disgusted - yellow
        elif(idx==1):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # fearful - blue
        elif(idx==2):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # happy - green
        elif(idx==3):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # neutral - black
        elif(idx==4):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # sad - orange
        elif(idx==5):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (128, 255, 255), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # surprised - cyan
        elif(idx==6):
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, f'{emotions[idx]}', (x+w//4, y-10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.imshow('image', img)
    else:
        pass

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''