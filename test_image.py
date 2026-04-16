from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("Litter.pt")

# Image path
image_path = r"ChatGPT Image Mar 11, 2026, 10_40_26 AM.png"

# Run prediction
results = model.predict(source=image_path, conf=0.25)

# Plot results
img = results[0].plot()

# Show image
cv2.imshow("Garbage Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()