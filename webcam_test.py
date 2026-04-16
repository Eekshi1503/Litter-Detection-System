from ultralytics import YOLO
import cv2

# Load model
model = YOLO("litteryolov8s.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.4,verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Garbage Detection Webcam", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()