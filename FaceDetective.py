import os
import cv2
import time
import matplotlib.pyplot as plt
# from FaceClassifer import *
import numpy as np

net = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                               caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def cvDnnDetectFaces(image, net, min_confidence=0.8):

    image_height, image_width, _ = image.shape
    output_image = image.copy()
    blob = cv2.dnn.blobFromImage(output_image, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    results = net.forward()
    padding = 0
    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * image_width)-padding
            y1 = int(bbox[1] * image_height)-padding
            x2 = int(bbox[2] * image_width)+padding
            y2 = int(bbox[3] * image_height)+padding
            cv2.rectangle(output_image, pt1=(x1, y1), pt2=(
                x2, y2), color=(0, 255, 0), thickness=image_width//200)

            cv2.putText(output_image, text=str(round(face_confidence, 3)), org=(x1, y1-25),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//600,
                        color=(255, 255, 255), thickness=image_width//200)

    return output_image


def dectectFace():
    capture = cv2.VideoCapture(0)
    i = 0
    while True:
        try:
            time.sleep(1/25)
            ret, frame = capture.read()
            if (not ret):
                break
            frame = cv2.resize(frame, (640, 480))
            newFrame = cvDnnDetectFaces(frame, net)
            cv2.imwrite('facedetect.jpg', newFrame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('facedetect.jpg', 'rb').read() + b'\r\n')

        except Exception as e:
            print(f'exc: {e}')
            pass

# dropFace(1000,0.4,True)
