import tkinter as tk
import numpy as np
import cv2

#Exit
def txtexitQ(frame):
    font=cv2.FONT_HERSHEY_SIMPLEX # font to use
    cv2.putText(frame,'Press q to exit!',(0,400),font,1,(0,0,255),2,cv2.LINE_AA)


# Funcion for buttons
def button1_click():
        print("button 1 is press")
        cap = cv2.VideoCapture(0)   #We use the camera number 0 of our computer

        #This is a big xml that has undreds of faces ja
        face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

        while True:
            ret, frame=cap.read()   #read the video

            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #convert to gray scale
            faces=face_cascade.detectMultiScale(gray,1.3,5) #detect the faces on  the gray image
            txtexitQ(frame)                                 #Just put the instruccion to exit

            #Create a for to track the size of the face and the location using a rectangle
            for(x,y,w,h) in faces:      
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
                roi_gray=gray[y:y+w,x:x+w]
                roi_color=frame[y:y+h,x:x+w]


            cv2.imshow('frame', frame)  #show the frame
            if cv2.waitKey(1)==ord('q'):    #press q to exit
                break

        cap.release()       #close our camera
        cv2.destroyAllWindows() #destroy the cv2 windows
        print("Face track finish") 
   
#Button to show only face
def button2_click():
    print("button 2 is press")
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create a mask of the same size of the frame
        mask = np.zeros_like(frame)

        for (x, y, w, h) in faces:
            # create a mask for the face
            face_mask = np.zeros_like(frame)
            cv2.rectangle(face_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

            # aply the mask face to the original frame
            face_only = cv2.bitwise_and(frame, face_mask)

            # Put the face in the black background
            mask = np.zeros_like(frame)
            mask[y:y + h, x:x + w] = face_only[y:y + h, x:x + w]
            txtexitQ(mask)

        cv2.imshow('frame', mask)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def button3_click():
    print("button 3 is press")

    cap = cv2.VideoCapture(0) #use to select our camera

    while True:
        ret, frame=cap.read()   #Read the frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width=int(cap.get(3))
        height=int(cap.get(4))

        corners = cv2.goodFeaturesToTrack(gray, 100, 0.4, 10)
        #                    (image, number of corners, minimun quality, distance)
        corners=np.int0(corners) #convert values float to int

        for corner in corners:
            x,y=corner.ravel()
            #ravel removes brakets inside [[1,2],[2,1]]-->[1,2,2,1]
            cv2.circle(frame,(x,y),5,(255,0,0),-1) #create the circles in the center of the corner
        txtexitQ(frame)
        cv2.imshow('frame',frame)#show in the frame, the image of zeros

        if cv2.waitKey(1)==ord('q'): #if we press the Q key break the program
            break

    cap.release()   #means that the camera can use in other program
    cv2.destroyAllWindows() #Just close all


def button4_click():
    #close
    print("Button 4 is press")
    window.destroy()

# make window
window = tk.Tk()

# size of window
window.geometry("400x300")  # weight x height

# Title windows
window.title("CV2 BASE")

# Title in the window
titulo = tk.Label(window, text="CV2 BASE", font=("Arial", 16))
titulo.pack()

# Make buttons
button1 = tk.Button(window, text="Detect face and tack", width=20, height=3, command=button1_click)
button1.pack()

button2 = tk.Button(window, text="Show face", width=20, height=3, command=button2_click)
button2.pack()

button3 = tk.Button(window, text="Detect corners", width=20, height=3, command=button3_click)
button3.pack()

button4 = tk.Button(window, text="exit", width=20, height=3, command=button4_click)
button4.pack()

# LOOP WINDOW
window.mainloop()