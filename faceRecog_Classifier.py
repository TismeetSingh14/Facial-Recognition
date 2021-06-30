import numpy as np
import os
import cv2

# KNN
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


cap = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
faceData = []
datasetPath = 'C:/Users/mande/Desktop/pep/dsml/machinelearning/facerecognition/'
label = []
class_id = 0
names = {}

# DATA PREPARATION

for fx in os.listdir(datasetPath):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data = np.load(datasetPath + fx)
        faceData.append(data)

        target = class_id*np.ones((data.shape[0],))
        class_id += 1
        label.append(target)

faceDataset = np.concatenate(faceData, axis=0)
faceLabel = np.concatenate(label, axis=0).reshape((-1, 1))

trainData = np.concatenate((faceDataset, faceLabel), axis=1)

# TESTING
while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    faces = faceCascade.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in faces:
        offset = 10
        faceSection = frame[y - offset:y + h +
                            offset, x - offset:x + w + offset]
        faceSection = cv2.resize(faceSection, (100, 100))

        out = knn(trainData, faceSection.flatten())

        pred_name = names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()