import cv2
import numpy as np

# INIT CAMERA
cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
faceData = []
datasetPath = 'C:/Users/mande/Desktop/pep/dsml/machinelearning/facerecognition/'
fileName = input()

while True :

    ret, frame = cap.read()

    if ret == False :
        continue
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frame,1.3, 5) 
    faces = sorted(faces, key = lambda f:f[2]*f[3])
    # print(faces)

    for x,y,w,h in faces[-1:]:           # ONLY PICKING THE LAST FACE AS IT IS THE LARGEST
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255),2)

        # EXTRACT/ CROPPING REQUIRED PART
        # EXTRACTING THE 'REGION OF INTEREST'
        offset = 10
        faceSection = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        faceSection = cv2.resize(faceSection, (100, 100))

        skip += 1
        if skip % 10 == 0:
            faceData.append(faceSection)
            # cv2.imshow("Face Section", faceSection)
            print(len(faceData))

    cv2.imshow("Frame", frame)
    key_pressed = cv2.waitKey(1) & 0XFF   
    if key_pressed == ord('q') :
        break

# CONVERTING FACEDATA TO A NUMPY ARRAY
faceData = np.asarray(faceData)
faceData = faceData.reshape((faceData.shape[0], -1))
print(faceData.shape)

# SAVING THE DATA INTO FILE SYSTEM
np.save(datasetPath + fileName + ".npy", faceData)
cap.release()
cv2.destroyAllWindows()