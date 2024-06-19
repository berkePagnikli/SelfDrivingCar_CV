import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('med_img_800.pt')

# Determine image size to match the training process
img_size = 800

# Load the image
image_path = r'Inference_Images\1478019975685727611_jpg.rf.zOsnI9GUtglDuUSJq97l.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not open image.")
    exit()

# Resize the image
resized_image = cv2.resize(image, (img_size, img_size))

# Perform inference
results = model(resized_image)

# Process results
for result in results:
    for bbox in result.boxes:
        conf = bbox.conf[0].item()

        # Only include detections with a confidence rate of 0.7 or above
        if conf >= 0.70:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            x1 = int(x1 * image.shape[1] / img_size)
            y1 = int(y1 * image.shape[0] / img_size)
            x2 = int(x2 * image.shape[1] / img_size)
            y2 = int(y2 * image.shape[0] / img_size)
            cls = int(bbox.cls[0].item())
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Save and display the output image
output_path = 'Inference_Outputs\output_image_1.jpg'
cv2.imwrite(output_path, image)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
