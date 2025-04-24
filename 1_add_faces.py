# 1_add_faces.py
import cv2
import pickle
import numpy as np
import os
import time

# --- Configuration ---
DATA_DIR = 'data/'
FACES_PKL = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL = os.path.join(DATA_DIR, 'names.pkl')
CASCADE_PATH = os.path.join('cascades', 'haarcascade_frontalface_default.xml')
FACE_SIZE = (50, 50)
FRAMES_TO_CAPTURE = 50 # Reduced for quicker capture, adjust as needed
CAPTURE_DELAY_MS = 100 # Capture roughly every 100ms if face detected
FACE_CONFIDENCE_THRESHOLD = 1.3 # Haar cascade scaleFactor
MIN_NEIGHBORS = 5 # Haar cascade minNeighbors

# --- Helper Functions ---
def validate_aadhar(aadhar_str):
    """ Basic validation for Aadhar number (numeric, specific length). """
    return aadhar_str.isdigit() and len(aadhar_str) == 12 # Assuming 12 digits

def ensure_dir_exists(path):
    """ Create directory if it doesn't exist. """
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)

# --- Main Logic ---
ensure_dir_exists(DATA_DIR)

# Load face detector
if not os.path.exists(CASCADE_PATH):
    print(f"ERROR: Cascade file not found at {CASCADE_PATH}")
    exit()
facedetect = cv2.CascadeClassifier(CASCADE_PATH)

# Get Aadhar number with validation
while True:
    aadhar_no = input("Enter your 12-digit Aadhar number: ")
    if validate_aadhar(aadhar_no):
        break
    else:
        print("Invalid Aadhar number. Please enter 12 digits.")

print("Initializing camera...")
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("ERROR: Could not open video source.")
    exit()

faces_data = []
frame_count = 0
capture_count = 0
last_capture_time = time.time()

print("\n--- Face Capture ---")
print(f"Look at the camera. Capturing {FRAMES_TO_CAPTURE} face samples.")
print("Press 'q' to quit early.")

while capture_count < FRAMES_TO_CAPTURE:
    ret, frame = video.read()
    if not ret:
        print("ERROR: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, FACE_CONFIDENCE_THRESHOLD, MIN_NEIGHBORS)

    display_frame = frame.copy() # Draw on a copy

    if len(faces) == 1: # Process only if exactly one face is detected
        x, y, w, h = faces[0]
        crop_img = frame[y:y+h, x:x+w] # Use original frame for cropping
        resized_img = cv2.resize(crop_img, FACE_SIZE)

        # Capture frame based on delay
        current_time = time.time()
        if current_time - last_capture_time > CAPTURE_DELAY_MS / 1000.0:
            faces_data.append(resized_img)
            capture_count += 1
            last_capture_time = current_time
            print(f"Captured: {capture_count}/{FRAMES_TO_CAPTURE}")

        # Draw rectangle and counter on the display frame
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        feedback_text = f"Captured: {capture_count}/{FRAMES_TO_CAPTURE}"
        cv2.putText(display_frame, feedback_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 255, 50), 2)

    elif len(faces) > 1:
         cv2.putText(display_frame, "Multiple faces detected!", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    else:
         cv2.putText(display_frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


    cv2.imshow('Capture Face - Press q to quit', display_frame)
    k = cv2.waitKey(10) # Increased wait key for smoother display
    if k == ord('q'):
        print("Capture stopped by user.")
        break

video.release()
cv2.destroyAllWindows()

if capture_count < FRAMES_TO_CAPTURE // 2 : # Check if enough frames were captured
     print(f"\nWarning: Only {capture_count} frames captured. Data might be insufficient.")
     print("Data not saved.")
elif faces_data:
    print("\nProcessing captured data...")
    faces_data_np = np.asarray(faces_data)
    # Reshape for KNN: (n_samples, n_features) -> (FRAMES_TO_CAPTURE, 50*50*3)
    # Assuming color images (3 channels). If using gray, change 3 to 1.
    try:
         faces_data_np = faces_data_np.reshape(capture_count, -1)
         print(f"Data shape: {faces_data_np.shape}")

         # --- Append data to existing files ---
         # Names
         if os.path.exists(NAMES_PKL):
             with open(NAMES_PKL, 'rb') as f:
                 names = pickle.load(f)
         else:
             names = []
         names.extend([aadhar_no] * capture_count)
         with open(NAMES_PKL, 'wb') as f:
             pickle.dump(names, f)
         print(f"Names data saved/appended to {NAMES_PKL}")

         # Faces
         if os.path.exists(FACES_PKL):
             with open(FACES_PKL, 'rb') as f:
                 faces = pickle.load(f)
             # Ensure shapes are compatible before appending
             if faces.shape[1] == faces_data_np.shape[1]:
                  faces = np.append(faces, faces_data_np, axis=0)
             else:
                  print(f"ERROR: Shape mismatch! Existing faces {faces.shape}, new faces {faces_data_np.shape}. Cannot append.")
                  print("Saving new data separately as faces_data_new.pkl")
                  faces = faces_data_np # Overwrite with new data for this example, or save separately
                  with open('data/faces_data_new.pkl', 'wb') as f:
                        pickle.dump(faces_data_np, f)

         else:
             faces = faces_data_np

         with open(FACES_PKL, 'wb') as f:
             pickle.dump(faces, f)
         print(f"Faces data saved/appended to {FACES_PKL}")
         print("\nFace data collection complete!")

    except Exception as e:
        print(f"Error processing or saving data: {e}")

else:
    print("\nNo face data captured.")