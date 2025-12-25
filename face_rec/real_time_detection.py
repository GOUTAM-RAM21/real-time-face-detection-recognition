import cv2
import face_recognition
import numpy as np
import os

# -------------------------------
# Load encodings and class names
# -------------------------------
def load_encodings(encodings_path):
    encodings = []
    class_names = []

    # Check if folder exists
    if not os.path.exists(encodings_path):
        print("❌ Encodings folder not found!")
        return encodings, class_names

    for file in os.listdir(encodings_path):
        if file.endswith("_encoding.npy"):
            class_name = file.split("_")[0]
            encoding = np.load(os.path.join(encodings_path, file))
            encodings.append(encoding)
            class_names.append(class_name)

    return encodings, class_names


# -------------------------------
# Path for saved encodings
# -------------------------------
encodings_path = "faces"
known_encodings, class_names = load_encodings(encodings_path)

if len(known_encodings) == 0:
    print("❌ No face encodings found. Exiting...")
    exit()

print(f"✅ Loaded classes: {class_names}")

# -------------------------------
# Initialize camera
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

# -------------------------------
# Main loop
# -------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through detected faces
    for face_encoding, face_location in zip(face_encodings, face_locations):

        matches = face_recognition.compare_faces(
            known_encodings,
            face_encoding,
            tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = class_names[first_match_index].upper()

        # Unpack face location
        top, right, bottom, left = face_location

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw label background
        cv2.rectangle(
            frame,
            (left, bottom - 25),
            (right, bottom),
            (0, 255, 0),
            cv2.FILLED
        )

        # Put name text
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )

    # Show output
    cv2.imshow("Face Recognition - Press 'q' to quit", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()
