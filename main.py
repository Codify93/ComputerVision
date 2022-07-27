import cv2
import numpy as np


whT= 320
cap = cv2.VideoCapture(0)

classNames = []
classesFile = "coco.names"
confidenceThreshold = 0.5
nmsThreshold = 0.3

modelCfg = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelCfg,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confidenceThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append((float(confidence)))
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confidenceThreshold,nmsThreshold)
    print(indices)
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+w),(255,0,20),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,20),2)

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    #print(layerNames)
    #print(layerNames)
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    outputs = net.forward(outputNames)

    #print(net.getUnconnectedOutLayers())
    findObjects(outputs,img)
    cv2.imshow('Image',img)
    cv2.waitKey(1)