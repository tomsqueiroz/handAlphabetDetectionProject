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

def realTimeInferAndSave(img, net, detectionTime):

    imageHeight, imageWidth = img.shape[:2]

    results = getBoundingBoxes(img, net)

    handCounter = 1

    for detection in results:

        id, name, confidence, x, y, w, h = detection

        cropped_img = img[y-int(0.0025*imageHeight):y+h+int(0.0025*imageHeight), x-int(0.0025*imageWidth):x+w+int(0.0025*imageWidth)]

        resized_cropped_img = cv2.resize(cropped_img, (50, 50))

        imageName = 'handAt' + detectionTime + '.png'

        cv2.imwrite(imageName, resized_cropped_img)
    
        handCounter += 1

    print("Found " + str(handCounter - 1) + " hand(s) ")



def realTimeCaptureAndInfer():

    cap = cv2.VideoCapture(0)
    framerate = cap.get(5)
    framecount = 0
    globalFrameCount = 0
    imgNumber = 0
    net = setNetFromPreTrainedParameters("models/cross-hands.cfg", "models/cross-hands.weights")


    while(True):
        # Capture frame-by-frame
        success, image = cap.read()

        if success:
            cv2.imshow('image',image)
        else:
            break

        framecount += 1
        globalFrameCount += 1
        if (framecount == framerate/2):
            framecount = 0
            imageName = 'realTimeImages/' + 'imageNumber' + str(imgNumber) + '.jpg'
            cv2.imwrite(imageName, image)
            realTimeInferAndSave(image, net, str(globalFrameCount))

            

        # Check end of video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



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

    imageCounter = 0 

    for image in images:

        img = cv2.imread(image)

        imageHeight, imageWidth = img.shape[:2]

        results = getBoundingBoxes(img, net)

        handCounter = 1

        for detection in results:

            id, name, confidence, x, y, w, h = detection

            cropped_img = img[y-int(0.0025*imageHeight):y+h+int(0.0025*imageHeight), x-int(0.0025*imageWidth):x+w+int(0.0025*imageWidth)]

            resized_cropped_img = cv2.resize(cropped_img, (50, 50))

            imageName = 'image' + str(imageCounter) + '_handNumber' + str(handCounter) + '.png'

            cv2.imwrite(imageName, resized_cropped_img)
        
            handCounter += 1

        print("Found " + str(handCounter - 1) + " hand(s) in image number " + str(imageCounter))

        imageCounter += 1 




if __name__ == "__main__":

    realTimeCaptureAndInfer()

    #net = setNetFromPreTrainedParameters("models/cross-hands.cfg", "models/cross-hands.weights")

    #images = getStaticImages("testVideo/")
    
    
    #inferAndSaveBBCrop(images, net)

