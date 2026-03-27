    import face_recognition
    import numpy as np
    import os
    from ultralytics import YOLO

    def crowda():
        # Step 1: Load and Train YOLO Model on COCO8 Dataset
        model = YOLO("yolo11n.pt")
        model.train(
            data="coco8.yaml",  
            epochs=10,  # Increased epochs for better accuracy
            imgsz=1080,  
            device="cpu"
        )
        model.val()
        
        # Step 2: Load Image & Add Padding
        image_path = "moh.webp"  
        image = cv2.imread(image_path)
        added_pixels = 50
        image = cv2.copyMakeBorder(image, added_pixels, added_pixels, added_pixels, added_pixels, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 3: Run YOLO Person Detection
        results = model(image)
        person_detections = [d for d in results[0].boxes if d.cls == 0]
        print(f"Number of people detected: {len(person_detections)}")
        
        # Step 4: Load Known Faces from Dataset
        KNOWN_FACES_DIR = "project dataset"
        known_encodings = []
        known_names = []
        
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
            
            for filename in os.listdir(person_folder):
                img_path = os.path.join(person_folder, filename)
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
        
        # Step 5: Recognize Faces in Detected People
        for detection in person_detections:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Get bounding box
            cropped_face = rgb_image[y1:y2, x1:x2]  # Crop detected face
            
            # Detect faces in cropped image
            face_locations = face_recognition.face_locations(cropped_face)
            face_encodings = face_recognition.face_encodings(cropped_face, face_locations)
            
            name = "Unknown"
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
                    name = known_names[best_match_index]
            
            # Draw bounding box & name on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Step 6: Show Final Output
        cv2.imshow("YOLO + Face Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()