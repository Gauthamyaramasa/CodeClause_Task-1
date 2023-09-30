import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Load COCO class names
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Function to perform object detection
def detect_objects(image_path):
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Prepare the input image for YOLOv3
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass
    detections = net.forward(layer_names)

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)

                # Draw bounding box in red
                cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 0, 255), 2)

                # Print class label and confidence in the console
                label = f"{classes[class_id]}: {confidence:.2f}"
                print(label)

    # Show the image with detections
    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = input("Enter the path to the image you want to analyze: ")
    detect_objects(image_path)
