#First
'''
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Changed to YOLOv8 for consistency

count = 0
cap = cv2.VideoCapture('edited2.mp4')
track_history = defaultdict(lambda: [])

# Define routes and their counts
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

cv2.namedWindow('FRAME')

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Sets to keep track of which objects have been in which areas
area_A = set()
area_B = set()
area_C = set()
area_D = set()
area_E = set()
area_F = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True, verbose=False)  # Added verbose=False to reduce console output

    # Get the boxes and track IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id
        class_ids = results[0].boxes.cls.cpu().tolist()
        if track_ids is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = []
    else:
        boxes = []
        track_ids = []

    # Visualize the results on the frame
    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

    # Process each detected object
    for box, track_id, class_ids in zip(boxes, track_ids, class_ids):
        if class_ids == 0:
            continue
        x_center, y_center, width, height = box
        x_center = int(x_center)
        y_center = int(y_center)
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False)
        result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False)
        result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False)
        result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False)
        result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False)
        result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False)

        # Update area sets and route counts based on object movement
        if result_A > 0:
            if track_id in area_E:
                route_dict["EA"] += 1
            if track_id in area_C:
                route_dict["CA"] += 1
            area_A.add(track_id)
        if result_B > 0:
            area_B.add(track_id)
        if result_C > 0:
            if track_id in area_B:
                route_dict["BC"] += 1
            area_C.add(track_id)
        if result_D > 0:
            if track_id in area_E:
                route_dict["ED"] += 1
            area_D.add(track_id)
        if result_E > 0:
            area_E.add(track_id)
        if result_F > 0:
            if track_id in area_B:
                route_dict["BF"] += 1
            if track_id in area_C:
                route_dict["CF"] += 1
            area_F.add(track_id)

    # Label each area with its identifier
    cv2.putText(annotated_frame, "A", (207, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "B", (540, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "C", (22, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "D", (7, 335), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "E", (845, 332), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "F", (869, 131), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)

    # Display the count of objects in each area
    cv2.putText(annotated_frame, str(len(area_A)), (210, 38), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_B)), (805, 36), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_C)), (50, 61), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_D)), (239, 578), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_E)), (701, 587), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_F)), (943, 56), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    # Display the annotated frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts
for key, value in route_dict.items():
    print(f"Route {key}: {value}")
'''
#Second
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('edited2.mp4')
track_history = defaultdict(lambda: [])

# Define routes and their counts
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

# Define counters for vehicle types
vehicle_counts = {
    "car": 0,
    "bus": 0,
    "motorcycle": 0
}

# YOLOv8 class indices for cars, buses, motorcycles
class_map = {2: "car", 5: "bus", 3: "motorcycle"}

cv2.namedWindow('FRAME')

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Sets to keep track of which objects have been in which areas
area_A = set()
area_B = set()
area_C = set()
area_D = set()
area_E = set()
area_F = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes = []
        track_ids = []

    # Visualize the results on the frame
    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Update vehicle count
        vehicle_type = class_map[class_id]
        vehicle_counts[vehicle_type] += 1

        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False)
        result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False)
        result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False)
        result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False)
        result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False)
        result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False)

        # Update area sets and route counts based on object movement
        if result_A > 0:
            if track_id in area_E:
                route_dict["EA"] += 1
            if track_id in area_C:
                route_dict["CA"] += 1
            area_A.add(track_id)
        if result_B > 0:
            area_B.add(track_id)
        if result_C > 0:
            if track_id in area_B:
                route_dict["BC"] += 1
            area_C.add(track_id)
        if result_D > 0:
            if track_id in area_E:
                route_dict["ED"] += 1
            area_D.add(track_id)
        if result_E > 0:
            area_E.add(track_id)
        if result_F > 0:
            if track_id in area_B:
                route_dict["BF"] += 1
            if track_id in area_C:
                route_dict["CF"] += 1
            area_F.add(track_id)

    # Label each area with its identifier
    cv2.putText(annotated_frame, "A", (207, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "B", (540, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "C", (22, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "D", (7, 335), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "E", (845, 332), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)
    cv2.putText(annotated_frame, "F", (869, 131), cv2.FONT_HERSHEY_SIMPLEX, 1, (234, 123, 250), 2)

    # Display the count of objects in each area
    cv2.putText(annotated_frame, str(len(area_A)), (210, 38), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_B)), (805, 36), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_C)), (50, 61), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_D)), (239, 578), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_E)), (701, 587), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.putText(annotated_frame, str(len(area_F)), (943, 56), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    # Display the annotated frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts
for key, value in route_dict.items():
    print(f"Route {key}: {value}")

# Print the total number of cars, buses, and motorcycles
print("\nTotal vehicle counts:")
for vehicle, count in vehicle_counts.items():
    print(f"{vehicle.capitalize()}: {count}")
'''
#Third-car,motorcycle,buses
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('edited2.mp4')

# Define routes and their counts
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

# Define sets to track unique IDs for each vehicle type
tracked_vehicles = {
    "cars": set(),
    "buses": set(),
    "motorcycles": set(),
    "all": set()  # To track unique vehicles across all routes
}

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Sets to keep track of which objects have been in which areas
area_A = set()
area_B = set()
area_C = set()
area_D = set()
area_E = set()
area_F = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id
        class_ids = results[0].boxes.cls.cpu().tolist()

        # Check if track_ids is not None
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
        else:
            track_ids = []
    else:
        boxes, track_ids, class_ids = [], [], []

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        if track_id is None:
            continue

        track_id = int(track_id)
        x_center, y_center, width, height = box
        x_center = int(x_center)
        y_center = int(y_center)
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))

        # Draw the center point of the object
        cv2.circle(frame, bottom_left, 4, (0, 255, 0), -1)

        # Track the vehicle type
        if class_id == 2:  # Car
            tracked_vehicles["cars"].add(track_id)
        elif class_id == 5:  # Bus
            tracked_vehicles["buses"].add(track_id)
        elif class_id == 3:  # Motorcycle
            tracked_vehicles["motorcycles"].add(track_id)

        # Track the vehicle across the routes
        if track_id not in tracked_vehicles["all"]:
            tracked_vehicles["all"].add(track_id)

            result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False)
            result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False)
            result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False)
            result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False)
            result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False)
            result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False)

            if result_A > 0:
                if track_id in area_E:
                    route_dict["EA"] += 1
                if track_id in area_C:
                    route_dict["CA"] += 1
                area_A.add(track_id)
            if result_B > 0:
                area_B.add(track_id)
            if result_C > 0:
                if track_id in area_B:
                    route_dict["BC"] += 1
                area_C.add(track_id)
            if result_D > 0:
                if track_id in area_E:
                    route_dict["ED"] += 1
                area_D.add(track_id)
            if result_E > 0:
                area_E.add(track_id)
            if result_F > 0:
                if track_id in area_B:
                    route_dict["BF"] += 1
                if track_id in area_C:
                    route_dict["CF"] += 1
                area_F.add(track_id)

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts
for key, value in route_dict.items():
    print(f"Route {key}: {value}")

# Print the total vehicle counts by type
print("\nTotal vehicle counts:")
print(f"Cars: {len(tracked_vehicles['cars'])}")
print(f"Buses: {len(tracked_vehicles['buses'])}")
print(f"Motorcycles: {len(tracked_vehicles['motorcycles'])}")

# Verify that total vehicle types match the sum of route counts
total_detected_vehicles = sum(route_dict.values())
total_tracked_vehicles = len(tracked_vehicles["all"])
print(f"\nTotal vehicles detected in routes: {total_detected_vehicles}")
print(f"Total unique vehicles tracked: {total_tracked_vehicles}")
'''
#SUV+3 wheeler
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('highway.mp4')

# Track unique IDs for each vehicle type
tracked_vehicles = defaultdict(set)

# Define routes and their counts
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 8: "3-wheeler"}

cv2.namedWindow('FRAME')

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Sets to keep track of which objects have been in which areas
area_A = set()
area_B = set()
area_C = set()
area_D = set()
area_E = set()
area_F = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Visualize the results on the frame
    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Determine the vehicle type
        vehicle_type = class_map[class_id]
        
        # Track the unique vehicle
        tracked_vehicles[vehicle_type].add(track_id)

        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False)
        result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False)
        result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False)
        result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False)
        result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False)
        result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False)

        # Update area sets and route counts based on object movement
        if result_A > 0:
            if track_id in area_E:
                route_dict["EA"] += 1
            if track_id in area_C:
                route_dict["CA"] += 1
            area_A.add(track_id)
        if result_B > 0:
            area_B.add(track_id)
        if result_C > 0:
            if track_id in area_B:
                route_dict["BC"] += 1
            area_C.add(track_id)
        if result_D > 0:
            if track_id in area_E:
                route_dict["ED"] += 1
            area_D.add(track_id)
        if result_E > 0:
            area_E.add(track_id)
        if result_F > 0:
            if track_id in area_B:
                route_dict["BF"] += 1
            if track_id in area_C:
                route_dict["CF"] += 1
            area_F.add(track_id)

    # Display the annotated frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts
for key, value in route_dict.items():
    print(f"Route {key}: {value}")

# Print the total number of cars, buses, motorcycles, 3-wheelers, and SUVs
print("\nTotal vehicle counts:")
for vehicle_type, count in tracked_vehicles.items():
    print(f"{vehicle_type.capitalize()}: {len(count)}")
'''
#Equal Routes + Vehicles
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

#cap = cv2.VideoCapture('highway.mp4')
cap = cv2.VideoCapture('edited2.mp4')
car = [0] * 15
bus = [0] * 15
SUV = [0] * 15
motorcycle = [0] * 15
three_wheelers = [0] * 15


# Track unique IDs for each vehicle type
tracked_vehicles = defaultdict(set)

# Track unique IDs across all routes
route_vehicles = set()

# Define routes and their counts
route_dict = {
    "EA": set(),
    "CA": set(),
    "BC": set(),
    "ED": set(),
    "BF": set(),
    "CF": set(),
}

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 8: "3-wheeler"}

cv2.namedWindow('FRAME')

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Visualize the results on the frame
    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Determine the vehicle type
        vehicle_type = class_map[class_id]
        
        # Track the unique vehicle
        tracked_vehicles[vehicle_type].add(track_id)
        route_vehicles.add(track_id)

        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False)
        result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False)
        result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False)
        result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False)
        result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False)
        result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False)

        # Update area sets and route counts based on object movement
        if result_A > 0:
            if track_id in route_dict["EA"]:
                route_dict["EA"].add(track_id)
            if track_id in route_dict["CA"]:
                route_dict["CA"].add(track_id)
            route_dict["EA"].add(track_id)
        if result_B > 0:
            route_dict["BF"].add(track_id)
        if result_C > 0:
            if track_id in route_dict["BC"]:
                route_dict["BC"].add(track_id)
            route_dict["BC"].add(track_id)
        if result_D > 0:
            if track_id in route_dict["ED"]:
                route_dict["ED"].add(track_id)
            route_dict["ED"].add(track_id)
        if result_E > 0:
            route_dict["ED"].add(track_id)
        if result_F > 0:
            if track_id in route_dict["BF"]:
                route_dict["BF"].add(track_id)
            if track_id in route_dict["CF"]:
                route_dict["CF"].add(track_id)
            route_dict["CF"].add(track_id)

    # Display the annotated frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts
print("\nRoute counts:")
for key, value in route_dict.items():
    print(f"Route {key}: {len(value)}")

# Print the total number of cars, buses, motorcycles, 3-wheelers, and SUVs
print("\nTotal vehicle counts:")
for vehicle_type, count in tracked_vehicles.items():
    print(f"{vehicle_type.capitalize()}: {len(count)}")'''
#YOLO v8
'''
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load a more robust YOLOv8 model
model = YOLO('yolov8m.pt')  # or use 'yolov8l.pt'

cap = cv2.VideoCapture('edited2.mp4')
track_history = defaultdict(lambda: [])

# Define routes and their counts
route_dict = {
    "EA": 0,
    "CA": 0,
    "BC": 0,
    "ED": 0,
    "BF": 0,
    "CF": 0,
}

# Define counters for vehicle types
vehicle_counts = {
    "car": 0,
    "bus": 0,
    "motorcycle": 0,
    "suv": 0,
    "3-wheeler": 0
}

# YOLOv8 class indices for cars, buses, motorcycles, SUVs, and 3-wheelers
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 9: "3-wheeler"}

cv2.namedWindow('FRAME')

# Define areas as polylines with their coordinates
# (Areas and the rest of the code remain unchanged)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    # (Drawing code remains unchanged)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes = []
        track_ids = []

    # Visualize the results on the frame
    annotated_frame = results[0].plot() if len(results) > 0 else frame.copy()

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Update vehicle count
        vehicle_type = class_map[class_id]
        vehicle_counts[vehicle_type] += 1

        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        # (Area checking code remains unchanged)

        # Update area sets and route counts based on object movement
        # (Area and route update code remains unchanged)

    # Display the annotated frame
    cv2.imshow("FRAME", annotated_frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print the route counts and total vehicle counts
for key, value in route_dict.items():
    print(f"Route {key}: {value}")

print("\nTotal vehicle counts:")
for vehicle, count in vehicle_counts.items():
    print(f"{vehicle.capitalize()}: {count}")
'''

#Minute wise
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('highway.mp4')

# Initialize arrays for vehicle counts (15 minutes)
car = [0] * 15
bus = [0] * 15
SUV = [0] * 15
motorcycle = [0] * 15
three_wheelers = [0] * 15

# Track vehicle IDs per minute
tracked_ids_per_minute = defaultdict(set)

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 8: "3-wheeler"}

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Start time tracking
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Current minute
    elapsed_time = time.time() - start_time
    current_minute = int(elapsed_time // 60)
    
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
                if vehicle_type == "car":
                    car[current_minute] += 1
                elif vehicle_type == "bus":
                    bus[current_minute] += 1
                elif vehicle_type == "suv":
                    SUV[current_minute] += 1
                elif vehicle_type == "motorcycle":
                    motorcycle[current_minute] += 1
                elif vehicle_type == "3-wheeler":
                    three_wheelers[current_minute] += 1

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()

# Print the vehicle counts for each minute
for i in range(15):
    print(f"Minute {i + 1}: Cars: {car[i]}, Buses: {bus[i]}, SUVs: {SUV[i]}, Motorcycles: {motorcycle[i]}, 3-Wheelers: {three_wheelers[i]}")

'''
#high
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('high.mp4')

# Initialize arrays for vehicle counts (15 minutes)
car = [0] * 15
bus = [0] * 15
SUV = [0] * 15
motorcycle = [0] * 15
three_wheelers = [0] * 15

# Track vehicle IDs per minute
tracked_ids_per_minute = defaultdict(set)

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 3: "motorcycle", 7: "suv", 8: "3-wheeler"}

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (800, 105), (840, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(700, 300), (700, 598), (800, 594), (800, 300)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Start time tracking
start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Current minute
    elapsed_time = time.time() - start_time
    current_minute = int(elapsed_time // 60)
    
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
                if vehicle_type == "car":
                    car[current_minute] += 1
                elif vehicle_type == "bus":
                    bus[current_minute] += 1
                elif vehicle_type == "suv":
                    SUV[current_minute] += 1
                elif vehicle_type == "motorcycle":
                    motorcycle[current_minute] += 1
                elif vehicle_type == "3-wheeler":
                    three_wheelers[current_minute] += 1

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:  # Esc key to exit
        break

cap.release()
cv2.destroyAllWindows()

# Print the vehicle counts for each minute
for i in range(15):
    print(f"Minute {i + 1}: Cars: {car[i]}, Buses: {bus[i]}, SUVs: {SUV[i]}, Motorcycles: {motorcycle[i]}, 3-Wheelers: {three_wheelers[i]}")

'''
#CSV
import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('edited2.mp4')

# Vehicle counts for the total across all junctions
vehicle_counts_total = defaultdict(int)

# Track unique IDs for each vehicle type per junction
tracked_vehicles_per_junction = {
    "EA": defaultdict(set),
    "CA": defaultdict(set),
    "BC": defaultdict(set),
    "ED": defaultdict(set),
    "BF": defaultdict(set),
    "CF": defaultdict(set),
}

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 7: "truck", 3: "two-wheeler", 0: "lcv", 1: "bicycle", 8: "three-wheeler"}

# Define areas as polylines with their coordinates
areaA=[(207,77),(515,85),(507,103),(173,105)]
areaB=[(540,82),(778,105),(829,132),(527,107)]
areaC=[(22,122),(7,335),(85,335),(163,99)]
areaD=[(7,337),(6,593),(121,595),(130,339)]
areaE=[(845,334),(868,598),(1010,594),(1006,337)]
areaF=[(869,133),(998,154),(1014,302),(884,302)]
# Initialize window
cv2.namedWindow('FRAME')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Determine the vehicle type
        vehicle_type = class_map[class_id]
        
        # Track the unique vehicle for the total count
        if track_id not in vehicle_counts_total:
            vehicle_counts_total[vehicle_type] += 1
        
        # Process for each junction
        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))

        # Check if the center point is inside any area and update junction counts
        for junction, area in [("EA", areaA), ("CA", areaB), ("BC", areaC), ("ED", areaD), ("BF", areaE), ("CF", areaF)]:
            if cv2.pointPolygonTest(np.array(area, np.int32), bottom_left, False) > 0:
                if track_id not in tracked_vehicles_per_junction[junction][vehicle_type]:
                    tracked_vehicles_per_junction[junction][vehicle_type].add(track_id)

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Print total vehicle counts
print("Total vehicle counts across all junctions:")
for vehicle_type, count in vehicle_counts_total.items():
    print(f"{vehicle_type.capitalize()}: {count}")

# Print vehicle counts per junction
print("\nVehicle counts at each junction:")
for junction, vehicle_dict in tracked_vehicles_per_junction.items():
    print(f"Junction {junction}:")
    for vehicle_type in class_map.values():
        print(f"  {vehicle_type.capitalize()}: {len(vehicle_dict[vehicle_type])}")

#Append CSV file
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import csv

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('edited2.mp4')

# Vehicle counts for the total across all junctions
vehicle_counts_total = defaultdict(int)

# Track unique IDs for each vehicle type per junction
tracked_vehicles_per_junction = {
    "EA": defaultdict(set),
    "CA": defaultdict(set),
    "BC": defaultdict(set),
    "ED": defaultdict(set),
    "BF": defaultdict(set),
    "CF": defaultdict(set),
}

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 7: "truck", 3: "two-wheeler", 0: "lcv", 1: "bicycle", 8: "three-wheeler"}

# Define areas as polylines with their coordinates
areaA = [(207, 77), (515, 85), (507, 103), (173, 105)]
areaB = [(540, 82), (778, 105), (829, 132), (527, 107)]
areaC = [(22, 122), (7, 335), (85, 335), (163, 99)]
areaD = [(7, 337), (6, 593), (121, 595), (130, 339)]
areaE = [(845, 334), (868, 598), (1010, 594), (1006, 337)]
areaF = [(869, 133), (998, 154), (1014, 302), (884, 302)]

# Initialize window
cv2.namedWindow('FRAME')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    cv2.polylines(frame, [np.array(areaA, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaB, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaC, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaD, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaE, np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(frame, [np.array(areaF, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Determine the vehicle type
        vehicle_type = class_map[class_id]
        
        # Track the unique vehicle for the total count
        if track_id not in vehicle_counts_total:
            vehicle_counts_total[vehicle_type] += 1
        
        # Process for each junction
        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))

        # Check if the center point is inside any area and update junction counts
        for junction, area in [("EA", areaA), ("CA", areaB), ("BC", areaC), ("ED", areaD), ("BF", areaE), ("CF", areaF)]:
            if cv2.pointPolygonTest(np.array(area, np.int32), bottom_left, False) > 0:
                if track_id not in tracked_vehicles_per_junction[junction][vehicle_type]:
                    tracked_vehicles_per_junction[junction][vehicle_type].add(track_id)

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Prepare data for CSV
csv_data = [["Turning Patterns", "Car", "Bus", "Truck", "Three-Wheeler", "Two-Wheeler", "LCV", "Bicycle"]]
for junction in tracked_vehicles_per_junction.keys():
    row = [junction]
    for vehicle_type in ["car", "bus", "truck", "three-wheeler", "two-wheeler", "lcv", "bicycle"]:
        row.append(len(tracked_vehicles_per_junction[junction][vehicle_type]))
    csv_data.append(row)

# Save data to CSV file
csv_file_path = '/Users/anaghamadhusoodhana/Desktop/BengaluruMobilityHack-aldrin 2/vehicle_counts.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"Data saved to {csv_file_path}")'''

#CSV correct number
'''import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import csv

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture('edited2.mp4')

# Vehicle counts for the total across all junctions
vehicle_counts_total = defaultdict(int)

# Track unique IDs for each vehicle type per junction
tracked_vehicles_per_junction = {
    "EA": defaultdict(set),
    "CA": defaultdict(set),
    "BC": defaultdict(set),
    "ED": defaultdict(set),
    "BF": defaultdict(set),
    "CF": defaultdict(set),
}

# YOLOv8 class indices mapped to vehicle types
class_map = {2: "car", 5: "bus", 7: "truck", 3: "two-wheeler", 0: "lcv", 1: "bicycle", 8: "three-wheeler"}

# Define areas as polylines with their coordinates
areas = {
    "EA": [(207, 77), (515, 85), (507, 103), (173, 105)],
    "CA": [(540, 82), (778, 105), (829, 132), (527, 107)],
    "BC": [(22, 122), (7, 335), (85, 335), (163, 99)],
    "ED": [(7, 337), (6, 593), (121, 595), (130, 339)],
    "BF": [(845, 334), (868, 598), (1010, 594), (1006, 337)],
    "CF": [(869, 133), (998, 154), (1014, 302), (884, 302)]
}

# Initialize window
cv2.namedWindow('FRAME')

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Draw polylines for each area
    for junction, area in areas.items():
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 255), 2)

    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, verbose=False)

    # Get the boxes, track IDs, and class IDs
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
        class_ids = results[0].boxes.cls.cpu().tolist()
    else:
        boxes, track_ids, class_ids = [], [], []

    # Process each detected object
    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
        # Skip non-relevant classes
        if class_id not in class_map:
            continue
        
        # Determine the vehicle type
        vehicle_type = class_map[class_id]
        
        # Track the unique vehicle for the total count
        if track_id not in tracked_vehicles_per_junction["EA"][vehicle_type]:
            vehicle_counts_total[vehicle_type] += 1
        
        # Process for each junction
        x_center, y_center, width, height = box
        bottom_left = (int(x_center - width / 2), int(y_center + height / 2))

        # Check if the center point is inside any area and update junction counts
        for junction, area in areas.items():
            if cv2.pointPolygonTest(np.array(area, np.int32), bottom_left, False) > 0:
                if track_id not in tracked_vehicles_per_junction[junction][vehicle_type]:
                    tracked_vehicles_per_junction[junction][vehicle_type].add(track_id)
                    # Draw bounding box and ID for debugging
                    cv2.rectangle(frame, (int(x_center - width / 2), int(y_center - height / 2)),
                                  (int(x_center + width / 2), int(y_center + height / 2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{vehicle_type} ID:{track_id}", (int(x_center - width / 2), int(y_center - height / 2) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("FRAME", frame)
    if cv2.waitKey(6) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Prepare data for CSV
csv_data = [["Turning Patterns", "Car", "Bus", "Truck", "Three-Wheeler", "Two-Wheeler", "LCV", "Bicycle", "Total"]]
for junction in tracked_vehicles_per_junction.keys():
    row = [junction]
    for vehicle_type in ["car", "bus", "truck", "three-wheeler", "two-wheeler", "lcv", "bicycle"]:
        row.append(len(tracked_vehicles_per_junction[junction][vehicle_type]))
    row.append(sum(len(tracked_vehicles_per_junction[junction][vt]) for vt in ["car", "bus", "truck", "three-wheeler", "two-wheeler", "lcv", "bicycle"]))
    csv_data.append(row)

# Add total counts to CSV
total_row = ["Total"]
for vehicle_type in ["car", "bus", "truck", "three-wheeler", "two-wheeler", "lcv", "bicycle"]:
    total_row.append(vehicle_counts_total[vehicle_type])
csv_data.append(total_row)

# Save data to CSV file
csv_file_path = '/Users/anaghamadhusoodhana/Desktop/BengaluruMobilityHack-aldrin 2/vehicle_counts.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)

print(f"Data saved to {csv_file_path}")
'''
