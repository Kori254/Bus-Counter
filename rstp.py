import cv2


ip_camera_url = "http://192.168.1.100:8080/video"

# Open the video stream
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Unable to connect to the IP camera.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera stream.")
        break

    # Display the video frame
    cv2.imshow("IP Camera Stream", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
