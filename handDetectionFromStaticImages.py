import argparse
import glob
import os

import cv2
import numpy as np




def getNetFromPreTrained(netConfigurationFile, netWeightsFile):

    print("oi")



def getStaticImages(imagesPath):

    files = sorted(glob.glob("%s/*.jpg" % imagesPath))

    return files


def setNetFromPreTrainedParameters(netConfigFile, netWeightsFile):

    net = cv2.dnn.readNetFromDarknet(netConfigFile, netWeightsFile)

    return net



def getBoundingBoxes(image, net):

    minimumConfidence = 0.5
    threshold=0.3
    labels = ["hand"]

    imageHeight, imageWidth = image.shape[:2]

    layerNames = net.getLayerNames()
    layerName = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)


    layerOutputs = net.forward(layerName)
    

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > minimumConfidence:
                
                box = detection[0:4] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
                (centerX, centerY, width, height) = box.astype("int")
                
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, minimumConfidence, threshold)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            id = classIDs[i]
            confidence = confidences[i]

            results.append((id, labels[id], confidence, x, y, w, h))

    return results


def inferAndSaveBBCrop(images, net):

    for image in images:

        img = cv2.imread(image)

        imageHeight, imageWidth = img.shape[:2]

        results = getBoundingBoxes(img, net)

        for detection in results:

            id, name, confidence, x, y, w, h = detection

            cropped_img = img[y-int(0.05*imageHeight):y+h+int(0.05*imageHeight), x-int(0.05*imageWidth):x+w+int(0.05*imageWidth)]

            resized_cropped_img = cv2.resize(cropped_img, (50, 50))
            cv2.imshow('image', resized_cropped_img)

            cv2.waitKey(0)




if __name__ == "__main__":

    net = setNetFromPreTrainedParameters("models/cross-hands.cfg", "models/cross-hands.weights")

    images = getStaticImages("images/")
    
    inferAndSaveBBCrop(images, net)

