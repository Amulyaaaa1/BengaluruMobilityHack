# import cv2
# import torch
# import numpy as np
# from collections import defaultdict
# from ultralytics import YOLO
# import time
# import os
# from map_to_15_second_intervals import get_interval_key, map_down


# def get_vehicle_count(video, start_seconds):
#     cap = cv2.VideoCapture(video)
    
#     count = 0

#     track_history = defaultdict(lambda: [])
    
#     semi_final_counts = []
    
#     car_count = [0] * 60
#     bus_count = [0] * 60
#     bicycle_count = [0] * 60
#     motorcycle_count = [0] * 60
#     car_ids, bus_ids, bicycle_ids, motorcycle_ids = [], [], [], []
    
    
#     # Track vehicle IDs per minute
#     tracked_ids_per_15s = defaultdict(set)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break
#         count += 1
#         if count % 4 != 0:
#             continue

#         frame = cv2.resize(frame, (1020, 600))
        
#         # Run YOLOv8 tracking on the frame, persisting tracks between frames
#         results = model.track(frame, persist=True, verbose=False)  # Added verbose=False to reduce console output

#         # Get the boxes and track IDs
#         if len(results) > 0 and results[0].boxes is not None:
#             boxes = results[0].boxes.xywh.cpu().numpy()
#             track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
#             class_ids = results[0].boxes.cls.cpu().tolist()
#         else:
#             boxes, track_ids, class_ids = [], [], []
            
        
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

#         fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
#         frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame count
#         elapsed_time = frame_count / fps
#         current_minute, current_seconds = divmod(elapsed_time, 60)
#         current_minute = int(current_minute)
#         current_seconds = int(current_seconds)
#         current_seconds_15 = get_interval_key(current_seconds)
                
#         print(f"Minute: {current_minute}, Seconds: {current_seconds}, Seconds_15: {current_seconds_15}")

#         if current_minute < 15:
#             for box, track_id, class_id in zip(boxes, track_ids, class_ids):
#                 # Skip non-relevant classes
#                 if class_id not in class_map:
#                     continue
                
#                 # Determine the vehicle type
#                 vehicle_type = class_map[class_id]
                
                
                
#                 # Check if the vehicle ID has already been tracked this minute
#                 if track_id not in tracked_ids_per_15s[current_seconds_15]:
#                     tracked_ids_per_15s[current_seconds_15].add(track_id)
                    
#                     # Increment count based on vehicle type
#                     if (vehicle_type == "car" or vehicle_type == "suv") and track_id not in car_ids:
#                         car_count[map_down[current_seconds_15]]+=1
#                         car_ids.append(track_id)
#                     elif vehicle_type == "bicycle" and track_id not in bicycle_ids:
#                         bicycle_count[map_down[current_seconds_15]] += 1
#                         bicycle_ids.append(track_id)
#                     elif vehicle_type == "motorcycle" and track_id not in motorcycle_ids:
#                         motorcycle_count[map_down[current_seconds_15]] += 1
#                         motorcycle_ids.append(track_id)
#                     elif vehicle_type == "bus" and track_id not in bus_ids:
#                         bus_count[map_down[current_seconds_15]]+=1
#                         bus_ids.append(track_id)
#                     elif vehicle_type == "person" and track_id not in motorcycle_ids:
#                         motorcycle_count[map_down[current_seconds_15]] += 1
#                         motorcycle_ids.append(track_id)

#         # Display the annotated frame
#         # cv2.imshow("FRAME", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
            
#     # Print the vehicle counts for each minute
#     i=0
#     for j in range(15,915,15):
#         print(f"Minute: {(start_seconds+j)//60 }, Seconds: {start_seconds+j}, Cars: {car_count[i]}, Buses: {bus_count[i]}, Motorcycles: {motorcycle_count[i]}, Bicycles: {bicycle_count[i]}")
#         semi_final_counts.append([start_seconds+j, car_count[i], bus_count[i], motorcycle_count[i], bicycle_count[i]])
#         i+=1
    
#     return semi_final_counts
 
   

    
# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')  # Changed to YOLOv8 for consistency

# Final_counts = ["Seconds", "Cars", "Buses", "Motorcycles", "Bicycles"]

#     # YOLOv8 class indices mapped to vehicle types
# class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "person"}



# areaA = [(530, 38), (393, 45),(172, 108), (504, 103)]
# areaB = [ (736, 52), (551, 36),(527, 129),(815, 150)]
# areaC = [(2, 66), (167, 42), (147, 311), (5, 309)]
# areaD = [(5, 310), (163, 311), (161, 599), (5, 598)]
# areaE =[(683, 333), (854, 333), (866, 598), (697, 597)]
# areaF = [(825, 79), (960, 96), (975, 334), (827, 333)]

# Videos = []
# for file in os.listdir("2024-05-15"):
#     if file.endswith(".mp4"):
#         Videos.append(os.path.join("2024-05-15", file))
# cv2.namedWindow('FRAME')
# time_offset = 0
# Videos.sort()

# for video in Videos[:1]:
#     print(video)
#     semi_final_count = get_vehicle_count(video, time_offset)
#     time_offset += 900
#     print(semi_final_count)
#     Final_counts.append(semi_final_count)
    
# print(Final_counts)

'''
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
    
    # Now storing vehicle counts for every 15 seconds (4 intervals per minute)
    car_count = [0] * 60  # 15 * 4 = 60 intervals in 15 minutes
    bus_count = [0] * 60
    bicycle_count = [0] * 60
    motorcycle_count = [0] * 60
    car_ids, bus_ids, bicycle_ids, motorcycle_ids = [], [], [], []

    # Track vehicle IDs per 15-second interval
    tracked_ids_per_interval = defaultdict(set)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        count += 1
        if count % 4 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, verbose=False)

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
        elapsed_time = frame_count / fps  # Elapsed time in seconds
        current_interval = int(elapsed_time // 15)  # Get 15-second intervals
        real_time = (current_interval * 15) // 60 + start_minute  # Calculate the real time in minutes

        if current_interval < 60:  # Only process the first 15 minutes (60 * 15 seconds = 15 minutes)
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # Skip non-relevant classes
                if class_id not in class_map:
                    continue
                
                # Determine the vehicle type
                vehicle_type = class_map[class_id]
                
                # Check if the vehicle ID has already been tracked in this 15-second interval
                if track_id not in tracked_ids_per_interval[current_interval]:
                    tracked_ids_per_interval[current_interval].add(track_id)
                    
                    # Increment count based on vehicle type
                    if (vehicle_type == "car" or vehicle_type == "suv") and track_id not in car_ids:
                        car_count[current_interval] += 1
                        car_ids.append(track_id)
                    elif vehicle_type == "bicycle" and track_id not in bicycle_ids:
                        bicycle_count[current_interval] += 1
                        bicycle_ids.append(track_id)
                    elif vehicle_type == "motorcycle" and track_id not in motorcycle_ids:
                        motorcycle_count[current_interval] += 1
                        motorcycle_ids.append(track_id)
                    elif vehicle_type == "bus" and track_id not in bus_ids:
                        bus_count[current_interval] += 1
                        bus_ids.append(track_id)
                    elif vehicle_type == "person" and track_id not in motorcycle_ids:
                        motorcycle_count[current_interval] += 1
                        motorcycle_ids.append(track_id)

        # Display the annotated frame
        # cv2.imshow("FRAME", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
            
    # Print the vehicle counts for each 15-second interval
    for i in range(60):
        minute = start_minute + (i * 15) // 60  # Convert 15-second intervals into minutes
        second = (i * 15) % 60  # Calculate the second within the minute
        print(f"Minute {minute}:{(minute*60)+second:02}: Cars: {car_count[i]}, Buses: {bus_count[i]}, Motorcycles: {motorcycle_count[i]}, Bicycles: {bicycle_count[i]}")
        semi_final_counts.append([minute, second, car_count[i], bus_count[i], motorcycle_count[i], bicycle_count[i]])
    
    return semi_final_counts
 
   

    
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

Final_counts = ["Minute", "Second", "Cars", "Buses", "Motorcycles", "Bicycles"]

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "person"}

Videos = []
for file in os.listdir("2024-05-15"):
    if file.endswith(".mp4"):
        Videos.append(os.path.join("2024-05-15", file))
cv2.namedWindow('FRAME')
time_offset = 0
Videos.sort()

for video in Videos[:2]:
    print(video)
    semi_final_count = get_vehicle_count(video, time_offset)
    time_offset += 15
    print(semi_final_count)
    Final_counts.append(semi_final_count)
    
print(Final_counts)


'''


import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os

def get_vehicle_count(video, start_seconds):
    cap = cv2.VideoCapture(video)
    count = 0

    # Lists to keep track of vehicle counts for each 15-second interval
    car_count = [0] * 60
    bus_count = [0] * 60
    bicycle_count = [0] * 60
    motorcycle_count = [0] * 60
    car_ids, bus_ids, bicycle_ids, motorcycle_ids = [], [], [], []
    
    tracked_ids_per_15s = defaultdict(set)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        count += 1
        if count % 4 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        
        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True, verbose=False)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            class_ids = results[0].boxes.cls.cpu().tolist()
        else:
            boxes, track_ids, class_ids = [], [], []
            
        annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()
        
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame count
        elapsed_time = frame_count / fps
        current_seconds = int(elapsed_time)
        current_seconds_15 = current_seconds // 15  # Get the current 15-second interval
        
        if current_seconds_15 < 60:
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id not in class_map:
                    continue
                
                vehicle_type = class_map[class_id]
                
                if track_id not in tracked_ids_per_15s[current_seconds_15]:
                    tracked_ids_per_15s[current_seconds_15].add(track_id)
                    
                    if (vehicle_type == "car" or vehicle_type == "suv") and track_id not in car_ids:
                        car_count[current_seconds_15] += 1
                        car_ids.append(track_id)
                    elif vehicle_type == "bicycle" and track_id not in bicycle_ids:
                        bicycle_count[current_seconds_15] += 1
                        bicycle_ids.append(track_id)
                    elif vehicle_type == "motorcycle" and track_id not in motorcycle_ids:
                        motorcycle_count[current_seconds_15] += 1
                        motorcycle_ids.append(track_id)
                    elif vehicle_type == "bus" and track_id not in bus_ids:
                        bus_count[current_seconds_15] += 1
                        bus_ids.append(track_id)
                    elif vehicle_type == "person" and track_id not in motorcycle_ids:
                        motorcycle_count[current_seconds_15] += 1
                        motorcycle_ids.append(track_id)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    semi_final_counts = []
    for i in range(60):
        second_offset = start_seconds + (i * 15)
        semi_final_counts.append([second_offset, car_count[i], bus_count[i], motorcycle_count[i], bicycle_count[i]])
        print(f"Seconds: {second_offset}, Cars: {car_count[i]}, Buses: {bus_count[i]}, Motorcycles: {motorcycle_count[i]}, Bicycles: {bicycle_count[i]}")
    
    return semi_final_counts


# Load the YOLOv8 model
model = YOLO('best (2).pt')

Final_counts = [["Seconds", "Cars", "Buses", "Motorcycles", "Bicycles"]]

# YOLOv8 class indices mapped to vehicle types
# class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 1: "bicycle", 0: "person"}
class_map = {0: "bus", 1: "car", 2: "lcv", 3: "three-wheeler", 4: "two-wheeler", 5: "truck", 6: "vehicle"}

Videos = []
for file in os.listdir("2024-05-15"):
    if file.endswith(".mp4"):
        Videos.append(os.path.join("2024-05-15", file))
cv2.namedWindow('FRAME')
time_offset = 1800
Videos.sort()

time_offset = 0
for video in Videos[2:3]:
    print(video)
    semi_final_count = get_vehicle_count(video, time_offset)
    time_offset += 900  # Increment by 900 seconds (15 minutes)
    Final_counts.extend(semi_final_count)

print(Final_counts)





