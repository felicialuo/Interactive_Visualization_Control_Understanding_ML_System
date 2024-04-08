import cv2
import csv
from fractions import Fraction

def readLabel2color():
    with open("../realtime_object_detection_w_depth/unique_label2color.csv", mode="r") as file:
        reader = csv.reader(file)
        label2color = {}
        next(reader)
        for row in reader:
            index, label, R, G, B = row
            label2color[label] = (  int(B), int(G), int(R))
    return label2color

def drawObjectDetection(obj_det_csv_path, color_frame, ifDepth):
    label2color = readLabel2color()

    with open(obj_det_csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            label, confidence, left, top, right, bottom, center_dist = row
            confidence, left, top, right, bottom = float(confidence), int(left), int(top), int(right), int(bottom)
            center_dist = round(float(center_dist), 2)
            color = label2color[label]

            cv2.rectangle(color_frame, (left,top), (right,bottom), color, 1)
            label += f" {int(confidence * 100)}%"

            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv2.putText(color_frame, label,(left,top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            if ifDepth:
                distance_string = str(center_dist) + " meter away"
                cv2.putText(color_frame,distance_string,(left,top+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def drawCLIP(csv_path, frame, color=(0,0,255)):
    height = 260
    cv2.putText(frame, "Per Second Prediction:", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1),
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            height += 10
            label, confidence = row
            cv2.putText(frame, f"{label} {confidence}", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            

def drawVCLIP(csv_path, frame, color=(255,255,255)):
    height = 330
    cv2.putText(frame, "Per Minute Prediction:", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            height += 10
            label, confidence = row
            cv2.putText(frame, f"{label} {confidence}", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            

def drawSKLTACT(csv_path, frame, color=(255,0,0)):
    height = 400
    cv2.putText(frame, "Skeleton-based Prediction:", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    with open(csv_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            height += 10
            label, confidence = row
            cv2.putText(frame, f"{label} {confidence}", (5,height), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
       

