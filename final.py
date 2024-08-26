import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import json
from dictionary import * 
from camera_junctions import *

def Get_outputs(video, offset): 
    global count
    global car_count_15, bus_count_15, bicycle_count_15, motorcycle_count_15, auto_count_15, lcv_count_15, truck_count_15
    global car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids
    global route_dict, total_boxes, desired_coords, required_coords, boxes_possible
    
    global route_class_vehicle_count
    route_class_vehicle_count = {route: defaultdict(int) for route in route_dict.keys()}

    print(f"Processing video: {video}")
    print("car_count", car_count_15)
    print("required_coords", required_coords)
    print("desired_coords", desired_coords)
    print("total_boxes", total_boxes)
    print("boxes_possible", boxes_possible)
    for i in car_count_15.keys():
        print(i)
    
    cap = cv2.VideoCapture(video)   
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # count += 1
        # if count % 3 != 0:
        #     continue

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
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame count
        elapsed_time = frame_count / fps
        current_seconds = int(elapsed_time) 
        current_seconds_15 = current_seconds // 15  # Get the current 15-second interval

        
        if current_seconds_15 < 60:
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
                
                
                
                for i in range(len(total_boxes)):         
                    if globals()["result"+"_"+boxes_possible[i]] > 0:
                        if track_id not in globals()["area"+"_"+boxes_possible[i]]:
                            globals()["area"+"_"+boxes_possible[i]].add(track_id)
                            vehicle_type = class_map[class_id]
                            for j in range(len(boxes_possible)):
                                # if j != i and track_id in globals()["area"+"_"+boxes_possible[j]]:
                                if track_id in globals()["area"+"_"+boxes_possible[j]] and boxes_possible[j]+boxes_possible[i] in car_count_15.keys() :
                                    # globals()["area"+"_"+boxes_possible[j]].remove(track_id)
                                    route_dict[boxes_possible[j] + boxes_possible[i]] += 1
                                    route_class_vehicle_count[boxes_possible[j] + boxes_possible[i]][vehicle_type] += 1
                                    if vehicle_type == "car":
                                        car_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "bus":
                                        bus_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "bicycle":
                                        bicycle_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "two-wheeler":
                                        motorcycle_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "three-wheeler":
                                        auto_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "lcv":
                                        lcv_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    elif vehicle_type == "truck":
                                        truck_count_15[boxes_possible[j]+boxes_possible[i]][current_seconds_15] += 1
                                    
                                
                for i in range(len(total_boxes)):
                    cv2.putText(annotated_frame, boxes_possible[i], coordinates[camera_id][boxes_possible[i]][0], cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, str(len(globals()["area"+"_"+boxes_possible[i]])), coordinates[camera_id][boxes_possible[i]][1], cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
                    
                cv2.imshow("FRAME", annotated_frame)
        if cv2.waitKey(6) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print the route counts
    for key, value in route_dict.items():
        print(f"Route {key}: {value}")
        
    # After processing the frame, display the counts for each route
    for route, vehicle_class_counts in route_class_vehicle_count.items():
        print(f"Route {route}:")
        for vehicle_class, count in vehicle_class_counts.items():
            print(f"  {vehicle_class}: {count} vehicles")
                
# Load the YOLOv8 model
model = YOLO('best (2).pt')  # Changed to YOLOv8 for consistency

Videos=[]
with open("input.json", "r") as f:
    data = json.load(f)
    camera_id = list(data.keys())[0]
    video_path_1 = data[camera_id]["Vid_1"]
    video_path_2 = data[camera_id]["Vid_2"]
    Videos.append(video_path_1)
    # Videos.append(video_path_2)

global count
count = 0
# cap = cv2.VideoCapture('highway.mp4')
track_history = defaultdict(lambda: [])
global car_count_15, bus_count_15, bicycle_count_15, motorcycle_count_15, auto_count_15, lcv_count_15, truck_count_15 
car_count_15, bus_count_15, bicycle_count_15, motorcycle_count_15, auto_count_15, lcv_count_15, truck_count_15 = {}, {}, {}, {}, {}, {}, {}
# car_count = {junction: 0 for junction in coordinates.keys()} 
for junction in Dynamic_path_dict[camera_id]:
    car_count_15[junction] = [0]*120
    bus_count_15[junction] = [0]*120
    bicycle_count_15[junction] = [0]*120
    motorcycle_count_15[junction] = [0]*120
    auto_count_15[junction] = [0]*120
    lcv_count_15[junction] = [0]*120
    truck_count_15[junction] = [0]*120

# car_count = 0
# bus_count = 0
# bicycle_count = 0
# motorcycle_count = 0
# auto_count = 0
# lcv_count = 0
# truck_count = 0
global car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids
car_ids, bus_ids, bicycle_ids, motorcycle_ids, auto_ids, lcv_ids, truck_ids = [], [], [], [], [], [], []


# YOLOv8 class indices mapped to vehicle types
#  0: Bus
#   1: Car
#   2: LCV
#   3: Three-Wheeler
#   4: Two-Wheeler
#   5: truck
#   6: vehicle
# class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "auto", 4: "lcv", 6: "truck"}
class_map = {0: "bus", 1: "car", 2: "lcv", 3: "three-wheeler", 4: "two-wheeler", 5: "truck", 6: "bicycle"}

# Define routes and their counts
global route_dict
route_dict = {}




    
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

offset = 0

for video in Videos:
    Get_outputs(video, offset)
    offset = offset + 900
        



for junction in Dynamic_path_dict[camera_id]:
    print(f"Junction: {junction}")
    print(f"Car count: {car_count_15[junction]}")
    print(f"Bus count: {bus_count_15[junction]}")
    print(f"Bicycle count: {bicycle_count_15[junction]}")
    print(f"Motorcycle count: {motorcycle_count_15[junction]}")
    print(f"Auto count: {auto_count_15[junction]}")
    print(f"LCV count: {lcv_count_15[junction]}")
    print(f"Truck count: {truck_count_15[junction]}")
    print("\n")