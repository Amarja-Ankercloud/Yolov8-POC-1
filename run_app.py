from ultralytics import YOLO
import cv2
import math
import time
import json

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
with open('data.json', 'w') as f:
    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        object_counts = {}

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # Get current timestamp
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                # with open('data.json', 'w') as f:
                # json.dump(timestamp, f)
                # print("Timestamp --->",timestamp)
                #logger.info("Timestamp --->", timestamp)

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # with open('data.json', 'w') as f:
                # json.dump(confidence, f)
                # print("Confidence --->",confidence)
                # logger.info("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                # with open('data.json', 'w') as f:
                # json.dump(classNames[cls], f)
                # print("Class name -->", classNames[cls])
                # logger.info("Class name -->", classNames[cls])

                # Extract class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # Increment count for the detected object
                object_counts[class_name] = object_counts.get(class_name, 0) + 1

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                # with open('data.json', 'w') as f:
                json_output = {
                    "timestamp": timestamp,
                    "object_counts": object_counts
                }
                f.write(json.dumps(json_output) + "\n")

                # json.dump(timestamp, f)
                # json.dump(confidence, f)
                # json.dump(classNames[cls], f)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()