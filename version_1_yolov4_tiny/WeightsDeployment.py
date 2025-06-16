import cv2
import numpy as np
import urllib.request
import requests
import time
from urllib.error import URLError, HTTPError

# ESP32-CAM URL
url = 'http://192.168.144.188/cam-hi.jpg'

# YOLOv4-tiny model configuration and weights
modelConfig = 'yolov4-tiny-custom.cfg'
modelWeights = 'yolov4-tiny-custom_best.weights'

# YOLO parameters
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Telegram bot details
bot_token = '7004287032:AAEAyp9bytq-WoR_Wi7P9dDj-GQ3xUu6DXk'
chat_id = '1309287453'

# Load YOLO class names
classesfile = 'obj.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLOv4-tiny model
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Timer variables
detection_start_time = {}
detection_duration = 0.3  # 20 seconds threshold

def send_telegram_message(message):
    send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={message}'
    response = requests.get(send_text)
    return response.json()

def send_telegram_image(image_path):
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    files = {'photo': open(image_path, 'rb')}
    data = {'chat_id': chat_id}
    response = requests.post(url, files=files, data=data)
    return response.json()

def get_image_from_url(url):
    try:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)
        return im
    except (URLError, HTTPError) as e:
        print(f"Error retrieving image: {e}")
        return None
    except Exception as e:
        print(f"Incomplete read: {e}")
        return None

def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    detected_classes = {classname: False for classname in classNames}

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if len(indices) > 0:
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        for i in indices:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            detected_classes[classNames[classIds[i]]] = True

            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Start or update the detection timer
            class_name = classNames[classIds[i]]
            if not detection_start_time.get(class_name):
                detection_start_time[class_name] = time.time()
            elif time.time() - detection_start_time[class_name] >= detection_duration:
                # Send notification
                send_telegram_message(f'Detected: {class_name.upper()} with {int(confs[i] * 100)}% confidence')

                # Save the image and send it to Telegram
                detection_image = "detection_result.jpg"
                cv2.imwrite(detection_image, im)
                send_telegram_image(detection_image)

                # Reset the detection timer
                detection_start_time[class_name] = time.time()

    # Reset detection timers if the object is no longer detected
    for classname in list(detection_start_time.keys()):
        if not detected_classes[classname]:
            del detection_start_time[classname]

while True:
    im = get_image_from_url(url)
    if im is None:
        continue  # Skip processing if the image could not be retrieved

    blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    outputs = net.forward(outputNames)
    findObject(outputs, im)

    cv2.imshow('Image', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
