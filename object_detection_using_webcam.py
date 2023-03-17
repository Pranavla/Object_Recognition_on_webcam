import cv2
from ultralytics import YOLO
import supervision as sv

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

yolo = YOLO("yolov8m.pt")
box_annotator = sv.BoxAnnotator(thickness = 2,text_scale = 1)

while True:
    success, frame = cam.read()

    res = yolo(frame)[0]
    det = sv.Detections.from_yolov8(res)
    lab = [f"{yolo.model.names[class_id]} {confidence:0.2f}"
           for _,confidence,class_id, _ in det]
    
    frame = box_annotator.annotate(scene = frame,detections = det, labels = lab)

    cv2.imshow("yolov8", frame)

    if(cv2.waitKey(30)==27): 
        break