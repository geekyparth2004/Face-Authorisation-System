import cv2
import numpy as np
import os
import pickle

# File to store registered faces
DATA_FILE = "face_data.pkl"

# Load or initialize face data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def register_face():
    cap = cv2.VideoCapture(0)
    print("Please look at the camera for face detection...")

    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Take the first detected face
            face_roi = frame[y:y+h, x:x+w]
            cv2.imshow("Face Detected", face_roi)
            cv2.waitKey(1000)
            cap.release()
            cv2.destroyAllWindows()

            name = input("Enter your name: ")
            face_encoding = cv2.resize(face_roi, (100, 100)).flatten()  # Resize and flatten the face image
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            print(f"Thank you, {name}! You have been registered.")

            # Save the updated data
            with open(DATA_FILE, "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
            break
        else:
            cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def scan_face():
    cap = cv2.VideoCapture(0)
    print("Please look at the camera for face recognition...")

    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_encoding = cv2.resize(face_roi, (100, 100)).flatten()  # Resize and flatten the face image

            # Compare with known faces
            min_dist = float("inf")
            name = "Unknown"

            for i, known_encoding in enumerate(known_face_encodings):
                dist = np.linalg.norm(face_encoding - known_encoding)  # Euclidean distance
                if dist < min_dist:
                    min_dist = dist
                    name = known_face_names[i]

            if min_dist < 5000:  # Threshold for recognition
                print(f"Welcome, {name}!")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def delete_registration():
    if not known_face_names:
        print("No registrations found.")
        return

    print("Registered names:")
    for i, name in enumerate(known_face_names):
        print(f"{i + 1}. {name}")

    try:
        choice = int(input("Enter the number of the name you want to delete: ")) - 1
        if 0 <= choice < len(known_face_names):
            deleted_name = known_face_names.pop(choice)
            known_face_encodings.pop(choice)
            print(f"{deleted_name} has been deleted.")
            # Save the updated data
            with open(DATA_FILE, "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")

def main():
    while True:
        print("\nWelcome to the Face Recognition System!")
        print("1. Register Yourself")
        print("2. Scan Face")
        print("3. Delete Registrations")
        print("4. Exit")
        choice = input("Please choose an option: ")

        if choice == "1":
            register_face()
        elif choice == "2":
            scan_face()
        elif choice == "3":
            delete_registration()
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()