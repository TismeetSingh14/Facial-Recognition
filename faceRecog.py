import cv2

# READ A VIDEO STREAM FROM CAMERA (FRAME BY FRAME)
# HERE 0 IS THE ID, 0 REPRESENTS DEFAULT DEVICE(WEBCAM)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(grayFrame,1.3, 5) 
    # 1.3 IS THE SCALING FACTOR. IT TELLS HOW MUCH TO SCALE EACH PORTION OF THE IMAGE
    # 5 IS THE NUMBER OF NEIGHBOURS

    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0),2)
        
    cv2.imshow("Video Frame", frame)
    # cv2.imshow("Gray Frame", grayFrame)
    key_pressed = cv2.waitKey(1) & 0XFF  # 0XFF IS 11111111
    # IF WE 'AND' A 32 BIT NUMBER WITH AN 8 BIT NUMBER TO  GET AN 8 BIT NUMBER
    # WE COMPARE THE 8 BIT NUMBER GENERATED TO THE ASCII VALUE OF 'q'
    # ONCE 'q' IS PRESSED THE PROGRAM EXITS  
    if key_pressed == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()