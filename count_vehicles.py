import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time
import os

def get_vehicle_count(video, start_minute):
    cap = cv2.VideoCapture(video)
    
    count = 0

    track_history = defaultdict(lambda: [])
    
    semi_final_counts = []
    
    car_count = [0] * 15
    bus_count = [0] * 15
    bicycle_count = [0] * 15
    motorcycle_count = [0] * 15
    car_ids, bus_ids, bicycle_ids, motorcycle_ids = [], [], [], []

    
    # Track vehicle IDs per minute
    tracked_ids_per_minute = defaultdict(set)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        count += 1
        if count % 4 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        
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
        current_minute, seconds = divmod(elapsed_time, 60)
        current_minute = int(current_minute)
        real_time = current_minute + start_minute
        
        # print(real_time, current_minute)

        if current_minute < 15:
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # Skip non-relevant classes
                if class_id not in class_map:
                    continue
                
                # Determine the vehicle type
                vehicle_type = class_map[class_id]
                
                # Check if the vehicle ID has already been tracked this minute
                if track_id not in tracked_ids_per_minute[current_minute]:
                    tracked_ids_per_minute[current_minute].add(track_id)
                    
                    # Increment count based on vehicle type
                    if (vehicle_type == "car" or vehicle_type == "suv") and track_id not in car_ids:
                        car_count[current_minute]+=1
                        car_ids.append(track_id)
                    elif vehicle_type == "bicycle" and track_id not in bicycle_ids:
                        bicycle_count[current_minute] += 1
                        bicycle_ids.append(track_id)
                    elif vehicle_type == "motorcycle" and track_id not in motorcycle_ids:
                        motorcycle_count[current_minute] += 1
                        motorcycle_ids.append(track_id)
                    elif vehicle_type == "bus" and track_id not in bus_ids:
                        bus_count[current_minute]+=1
                        bus_ids.append(track_id)
                    elif vehicle_type == "person":
                        motorcycle_count[current_minute] += 1
                        motorcycle_ids.append(track_id)

        # Display the annotated frame
        # cv2.imshow("FRAME", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
            
    # Print the vehicle counts for each minute
    for i in range(15):
        print(f"Minute {real_time + i}: Cars: {car_count[i]}, Buses: {bus_count[i]}, Motorcycles: {motorcycle_count[i]}, Bicycles: {bicycle_count[i]}")
        semi_final_counts.append([real_time + i, car_count[i], bus_count[i], motorcycle_count[i], bicycle_count[i]])
    
    return semi_final_counts
 
   

    
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Changed to YOLOv8 for consistency

Final_counts = ["Minute", "Cars", "Buses", "Motorcycles", "Bicycles"]

    # YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "person"}



areaA = [(530, 38), (393, 45),(172, 108), (504, 103)]
areaB = [ (736, 52), (551, 36),(527, 129),(815, 150)]
areaC = [(2, 66), (167, 42), (147, 311), (5, 309)]
areaD = [(5, 310), (163, 311), (161, 599), (5, 598)]
areaE =[(683, 333), (854, 333), (866, 598), (697, 597)]
areaF = [(825, 79), (960, 96), (975, 334), (827, 333)]

Videos = []
for file in os.listdir("2024-05-15"):
    if file.endswith(".mp4"):
        Videos.append(os.path.join("2024-05-15", file))
cv2.namedWindow('FRAME')
time_offset = 0
Videos.sort()

for video in Videos[0:2]:
    print(video)
    semi_final_count = get_vehicle_count(video, time_offset)
    time_offset += 15
    print(semi_final_count)
    Final_counts.append(semi_final_count)
    
print(Final_counts)