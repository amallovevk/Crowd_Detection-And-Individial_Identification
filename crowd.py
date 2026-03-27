def crowdaa():
   import cv2
   import numpy as np
   from ultralytics import YOLO
   

   def train_yolo():
        model = YOLO("yolo11n.pt")
        image, detections = detect_people(model, "moh.webp")
        model.train(
            data="coco8.yaml",  
            epochs=1,  
            imgsz=1080,  
            device="cpu"
        )
        model.val()
        model.export(format="onnx")
        return model

   def detect_people(model, image_path):
        # Load image and add pixels
        image = cv2.imread(image_path)
        added_pixels = 50
        image = cv2.copyMakeBorder(image, added_pixels, added_pixels, added_pixels, added_pixels, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # YOLO detection
        results = model(rgb_image)
        person_detections = [detection for detection in results[0].boxes if detection.cls == 0]
        print(f"Number of people detected: {len(person_detections)}")

        # Return processed image and detected people
        return image, person_detections
   if __name__ == '__main__':
        model = train_yolo()
        np.save("detected_people.npy", detections)  # Save detections for next module
        image, detections = detect_people(model, "moh.webp")
        cv2.imwrite("processed_crowd.jpg", image)
