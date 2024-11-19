import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import numpy as np

model = YOLO('yolov8n.pt')

# Define the areas
area1 = np.array([(160,1), (342, 1), (338, 357), (142,355)], np.int32)
area2 = np.array([(348, 1), (545, 1), (536, 356), (341, 357)], np.int32)


# Initialize the tracker and set to track people
tracker = Tracker()

# Dictionaries to track people
people_entering_area1 = set()  # Set of people who entered area1
people_entering_area2 = set()  # Set of people who entered area2

# Open video capture
cap = cv2.VideoCapture('test.mp4')

# Load COCO class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 4 != 0:  # Process every 4th frame for speed
        continue

    frame = cv2.resize(frame, (640, 360))

    # Run inference
    results = model.predict(frame)
    detections = results[0].boxes.data

    print(f"Detections: {len(detections)}")

    detection_df = pd.DataFrame(detections).astype("float")

    tracked_bboxes = []

    # Filter out only people (class_id = 'person')
    for index, row in detection_df.iterrows():
        x1, y1, x2, y2, class_id = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[5])
        class_name = class_list[class_id]

        if class_name == 'person':  # Only track people
            tracked_bboxes.append([x1, y1, x2, y2])

    print(f"Person Bounding Boxes: {tracked_bboxes}")

    bbox_id_list = tracker.update(tracked_bboxes)
    print(f"Tracked Objects: {bbox_id_list}")

    for bbox in bbox_id_list:
        x3, y3, x4, y4, obj_id = bbox

        # Calculate the center of the bounding box
        center_x = (x3 + x4) // 2
        center_y = (y3 + y4) // 2

        print(f"Object ID: {obj_id}, Center Coordinates: ({center_x},{center_y})")

        # Check if the person enters area1 using the center point
        result_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (center_x, center_y), False)

        if result_area1 >= 0:
            # Add to people_entering_area1 if entering area 1
            people_entering_area1.add(obj_id)
            print(f"Object {obj_id} entered area1")

        # Check if the object has already entered area1 and is now in area2
        if obj_id in people_entering_area1:
            result_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (center_x, center_y), False)
            if result_area2 >= 0:
                # If the person has entered area2 after area1, add to area2 set
                people_entering_area2.add(obj_id)
                print(f"Object {obj_id} entered area2")

                # Draw bounding box and label the object
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255, 100, 255), -1)
                cv2.putText(frame, str(obj_id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # Draw areas on the frame
    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str(1), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str(2), (504, 471), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Display the count of people who entered both areas
    cv2.putText(frame, f"People Entering: {len(people_entering_area2)}", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("RGB", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
