import cv2
import face_recognition
import numpy as np
import pickle
import os

# File to store registered faces
DATA_FILE = "face_data.pkl"

# Load or initialize face data
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

def register_face():
    cap = cv2.VideoCapture(0)
    print("Please look at the camera for face detection...")

    while True:
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            cv2.imshow("Face Detected", frame)
            cv2.waitKey(1000)
            cap.release()
            cv2.destroyAllWindows()

            name = input("Enter your name: ")
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(f"Welcome, {name}!")
                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Display the name
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

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