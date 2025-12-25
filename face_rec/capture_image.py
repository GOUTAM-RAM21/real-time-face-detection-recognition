import cv2
import face_recognition
import os
import numpy as np

# ===============================
# Create folder to store faces
# ===============================
os.makedirs("faces", exist_ok=True)

# ===============================
# Get person name
# ===============================
name = input("Enter name: ").strip()
if name == "":
    print("Name cannot be empty")
    exit()

# ===============================
# Initialize webcam
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Press 'c' to capture image")
print("Press 'q' to quit")

# ===============================
# Main loop
# ===============================
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Show live frame
    cv2.imshow("Training - Press 'c' to capture | 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    # ===============================
    # Capture image
    # ===============================
    if key == ord('c'):
        img_path = f"faces/{name}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Image saved at: {img_path}")

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get face encodings
        encodings = face_recognition.face_encodings(rgb_img)

        if len(encodings) > 0:
            np.save(f"faces/{name}_encoding.npy", encodings[0])
            print(f"Encoding saved for {name}")
        else:
            print("No face detected. Try again.")

    # ===============================
    # Quit
    # ===============================
    if key == ord('q'):
        break

# ===============================
# Release resources
# ===============================
cap.release()
cv2.destroyAllWindows()
