import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Changed to YOLOv8 for consistency

count = 0
cap = cv2.VideoCapture('highway.mp4')
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
    if count % 4 != 0:
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
        center = (x_center, y_center)
        track_id = int(track_id)

        # Draw the center point of the object
        cv2.circle(annotated_frame, bottom_left, 4, (0, 255, 0), -1)
        cv2.circle(annotated_frame, center, 4, (0, 255, 0), -1)

        # Check if the center point is inside any area
        result_A = cv2.pointPolygonTest(np.array(areaA, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaA, np.int32), center, False)
        result_B = cv2.pointPolygonTest(np.array(areaB, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaB, np.int32), center, False)
        result_C = cv2.pointPolygonTest(np.array(areaC, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaC, np.int32), center, False) 
        result_D = cv2.pointPolygonTest(np.array(areaD, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaD, np.int32), center, False)
        result_E = cv2.pointPolygonTest(np.array(areaE, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaE, np.int32), center, False)
        result_F = cv2.pointPolygonTest(np.array(areaF, np.int32), bottom_left, False) or cv2.pointPolygonTest(np.array(areaF, np.int32), center, False)
    

        # Update area sets and route counts based on object movement
        if result_A > 0:
            if track_id in area_E or track_id in area_F:
                route_dict["EA"] += 1
            if track_id in area_C:
                route_dict["CA"] += 1
            # area_A.add(track_id)
        if result_B > 0:
            area_B.add(track_id)
        if result_C > 0:
            if track_id in area_B:
                route_dict["BC"] += 1
            area_C.add(track_id)
        if result_D > 0:
            if track_id in area_E or track_id in area_F:
                route_dict["ED"] += 1
            if track_id in area_B:
                route_dict["BC"] += 1
            # area_D.add(track_id)
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