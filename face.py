import cv2
import pickle
import numpy as np
from deepface import DeepFace

ENCODINGS_FILE = "face_encodings.pkl"

def recognize_faces(image_path):
    # Load trained encodings
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    # Read input image
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    detections = DeepFace.detectFace(image_path, detector_backend="opencv")  # You can change backend ("mtcnn", "retinaface", etc.)
    
    if detections is None:
        print("No faces detected.")
        return

    for i, detected_face in enumerate(detections):
        try:
            # Get face embedding
            embedding = DeepFace.represent(detected_face, model_name="Facenet")

            # Find the best match
            distances = [np.linalg.norm(np.array(embedding) - np.array(enc)) for enc in known_encodings]
            best_match_index = np.argmin(distances)
            
            if distances[best_match_index] < 10:  # Adjust threshold as needed
                name = known_names[best_match_index]
            else:
                name = "Unknown"

            # Draw bounding box and label (for simplicity, assume one face detected)
            x, y, w, h = 50, 50, 100, 100  # Replace with real face bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error recognizing face: {e}")

    # Show the image
    cv2.imshow("Recognized Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces("crowd.jpg")  # Change to your input image
