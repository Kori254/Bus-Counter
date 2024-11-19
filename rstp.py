import cv2
from ultralytics import YOLO

# Load YOLO model (you can use 'yolov5s.pt' or a custom-trained model)
model = YOLO('yolov8n.pt')  # Replace with your model's path

# Replace 'rtsp://...' with your RTSP stream URL 
#rtsp://[username]:[password]@[IP address]:554/live

rtsp_url = 'rtsp://Kori:Kori1234@172.167.0.2:554/live'

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

# Process the video stream frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break
    
    # Perform object detection
    results = model.predict(frame)

    # Draw the bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes:
            # Extract the box coordinates and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLO RTSP Stream', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
