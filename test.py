import cv2                                               #python library which can be used to solve computer vision problems
import os                                                #module which provide functions for interacting with operating system
from keras.models import load_model                      #high level API used to built machine learning models
import numpy as  np                                      #python library used for working with arrays
from pygame import mixer                                 #library used to play audio file through mixer module
import time

mixer.init()  #ACTIVATING MIXER
sound = mixer.Sound('alarm.wav') 

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['CLOSE','OPEN']

model = load_model('models/cnncat2.h5')
path = os.getcwd()  
cap = cv2.VideoCapture(0) #accessing camera
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0 #timer
thicc=2
rpred=[99]
lpred=[99]

while(True):  #INFINITE LOOP TO CAPTURE EACH FRAME
    ret, frame = cap.read()      #reading each frame and storing image in frame var
    height,width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #OpenCV algo takes grey image as input so covert img to grey

    #face and eye detection
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25)) 
    left_eye = leye.detectMultiScale(gray) 
    right_eye = reye.detectMultiScale(gray)
    
    cv2.rectangle(frame, (0,height-50) , (300,height) , (0,0,0) , thickness=cv2.FILLED )
    

    #creating ROI

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (255,255,0) , 3 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w] #feeding right eye to CNN classifier 
        count=count+1
      #correcting dimensions of right eye image   
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))   #model is trained on 24*24pixels
        r_eye= r_eye/255  #better convergence (0-255 color scale,rgb) 
        r_eye= r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred =np.argmax(model.predict(r_eye),axis=-1) #classifier ready to predict
        if(rpred[0]==1):
            lbl='Open'
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]    #feeding left eye to CNN classifier
        count=count+1
        #correcting dimensions of left eye image
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye,(24,24))  #model is trained on 24*24pixels
        l_eye= l_eye/255    #better convergence (0-255 color scale,rgb)
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred =np.argmax(model.predict(l_eye),axis=-1)
        if(lpred[0]==1):
            lbl='Open'
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):    #if both eyes closed
        score=score+1
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 6 )
        for (x,y,w,h) in right_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 1 )
        for (x,y,w,h) in left_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 1 )
        cv2.putText(frame,'DROWSY!!',(150,height-440), font, 3.0,(0,0,255),3,cv2.LINE_AA)
        cv2.putText(frame,"EYES STATUS: CLOSED",(10,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)  #PUTTING TEXT
        cv2.rectangle(frame, (0,height-50) , (300,height) , (0,0,255) , 2)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 6 )
        for (x,y,w,h) in right_eye:
             cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        for (x,y,w,h) in left_eye:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1)
        cv2.putText(frame,'ACTIVE',(150,height-440), font, 3.0,(0,255,0),3,cv2.LINE_AA)
        cv2.putText(frame,"EYES STATUS: OPEN",(10,height-20), font, 1,(0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(frame, (0,height-50) , (300,height) , (0,255,0) , 2)
    if(score<0):
        score=0
    cv2.putText(frame,'SCORE',(5,height-400), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(frame,str(score),(15,height-350), font, 3.0,(255,255,255),3,cv2.LINE_AA)
    if(score>=5):
        #DANGER CONDITION
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play() #play alarm

        except: 
            pass
        if(thicc<6):#thickness of red border
            thicc= thicc+10
        else:
            thicc=thicc-10
            if(thicc<10):
                thicc=10
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
       
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
