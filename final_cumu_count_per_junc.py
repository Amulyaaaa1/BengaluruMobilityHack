import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import json
from coordinates import * 

def Get_outputs(video): 
    global count
    global car_count, bus_count, bicycle_count, motorcycle_count, auto_count, lcv_count, truck_count
    global car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids
    global route_dict, total_boxes, desired_coords, required_coords, boxes_possible
    
    print(f"Processing video: {video}")
    print("car_count", car_count)
    print("required_coords", required_coords)
    print("desired_coords", desired_coords)
    print("total_boxes", total_boxes)
    print("boxes_possible", boxes_possible)
    
    cap = cv2.VideoCapture(video)   
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))

        # Draw polylines for each area
        for i in range(len(required_coords.keys())):
            cv2.polylines(frame, [np.array(required_coords[list(required_coords.keys())[i]], np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
        # cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)  # Added verbose=False to reduce console output

        # Get the boxes and track IDs
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            class_ids = results[0].boxes.cls.cpu().tolist()
        else:
            boxes, track_ids, class_ids = [], [], []
            
        # Visualize the results on the frame
        annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            # Skip non-relevant classes
            if class_id not in class_map:
                continue
            
            # Determine the vehicle type
            vehicle_type = class_map[class_id]
            

            x_center, y_center, width, height = box
            x_center = int(x_center)
            y_center = int(y_center)
            bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
            center = (x_center, y_center)
            track_id = int(track_id)

            # Draw the center point of the object
            cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)
            cv2.circle(annotated_frame, center, 4, (0, 255, 0), -1)
        
            for i in range(len(total_boxes)):
                globals()["result"+"_"+boxes_possible[i]] = cv2.pointPolygonTest(np.array(required_coords[list(required_coords.keys())[i]], np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(required_coords[list(required_coords.keys())[i]], np.int32), center, False)
            
            
            # for i in range(len(total_boxes)):
            #     for j in range(i+1, len(total_boxes)):
            #         if globals()["result"+"_"+boxes_possible[i]]>0:
            #             globals()["area"+"_"+boxes_possible[i]].add(track_id)
            #             if track_id in globals()["area"+"_"+boxes_possible[j]]:
            #                 # globals()["area"+"_"+boxes_possible[j]].remove(track_id)
            #                 route_dict[boxes_possible[j]+boxes_possible[i]] += 1
            for i in range(len(total_boxes)):         
                if globals()["result"+"_"+boxes_possible[i]]>0 and track_id not in globals()["area"+"_"+boxes_possible[i]]:
                    globals()["area"+"_"+boxes_possible[i]].add(track_id)
                    other_boxes = [box for box in boxes_possible if box != boxes_possible[i]]
                    for j in range(len(other_boxes)):
                        if track_id in globals()["area"+"_"+other_boxes[j]]:
                            # globals()["area"+"_"+other_boxes[j]].remove(track_id)
                            route_dict[boxes_possible[j]+other_boxes[i]] += 1
                            
            # for i in range(len(total_boxes)):
            #     cv2.putText(annotated_frame, total_boxes[i], coordinates[camera_id][boxes_possible[i]][0], cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
            #     cv2.putText(annotated_frame, str(len(globals()["area"+"_"+boxes_possible[i]])), coordinates[camera_id][boxes_possible[i]][1], cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                
            cv2.imshow("FRAME", annotated_frame)
        if cv2.waitKey(6) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the route counts
    for key, value in route_dict.items():
        print(f"Route {key}: {value}")
                
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Changed to YOLOv8 for consistency

global count
count = 0
cap = cv2.VideoCapture('highway.mp4')
track_history = defaultdict(lambda: [])
global car_count, bus_count, bicycle_count, motorcycle_count, auto_count, lcv_count, truck_count
car_count = 0
bus_count = 0
bicycle_count = 0
motorcycle_count = 0
auto_count = 0
lcv_count = 0
truck_count = 0
global car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids
car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids = [], [], [], [], [], [], []


# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "auto", 4: "lcv", 6: "truck"}

# Define routes and their counts
global route_dict
route_dict = {}
Videos=[]


with open("input.json", "r") as f:
    data = json.load(f)
    camera_id = list(data.keys())[0]
    video_path_1 = data[camera_id]["Vid_1"]
    video_path_2 = data[camera_id]["Vid_2"]
    Videos.append(video_path_1)
    # Videos.append(video_path_2)
    
global boxes_possible, required_coords, desired_coords, total_boxes
boxes_possible = list(coordinates[camera_id].keys()) #['A', 'B', 'C', 'D', 'E', 'F']

# #permuatations of routes possible from the given coordinates
for i in range(len(boxes_possible)):
    for j in range(i+1, len(boxes_possible)):
        if str(boxes_possible[i]+boxes_possible[j]) not in route_dict:
            route_dict[boxes_possible[i]+boxes_possible[j]] = 0
        if str(boxes_possible[j]+boxes_possible[i]) not in route_dict:
            route_dict[boxes_possible[j]+boxes_possible[i]] = 0




cv2.namedWindow('FRAME')

# areaA = [(530, 38), (393, 45),(172, 108), (504, 103)]
# areaB = [ (736, 52), (551, 36),(527, 129),(815, 150)]
# areaC = [(2, 66), (167, 42), (147, 311), (5, 309)]
# areaD = [(5, 310), (163, 311), (161, 599), (5, 598)]
# areaE = [(683, 333), (854, 333), (866, 598), (697, 597)]
# areaF = [(825, 79), (960, 96), (975, 334), (827, 333)]


desired_coords = coordinates[camera_id]
required_coords = {}
for box, coords in desired_coords.items():
    # required_coords[str("area"+box)] = coords
    globals()["area"+box] = coords
    required_coords[box] = coords

    
# Sets to keep track of which objects have been in which areas
# area_A = set()
# area_B = set()
# area_C = set()
# area_D = set()
# area_E = set()
# area_F = set()
total_boxes=[]
for box in required_coords:
    globals()["area"+"_"+box] = set()
    total_boxes.append(globals()["area"+"_"+box])


for video in Videos:
    Get_outputs(video)
        