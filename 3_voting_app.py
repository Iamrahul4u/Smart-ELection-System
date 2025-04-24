# 3_voting_app.py
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import threading # To prevent GUI freezing during TTS

# --- Optional: Cross-platform Text-to-Speech ---
try:
    import pyttsx3
    engine = pyttsx3.init()
except ImportError:
    print("Warning: pyttsx3 not found. Text-to-speech disabled.")
    print("Install using: pip install pyttsx3")
    engine = None

def speak(text):
    """ Speaks the given text using pyttsx3 if available. Runs in a separate thread. """
    if engine:
        try:
            # Run speak in a non-blocking thread
            threading.Thread(target=engine.say, args=(text,), daemon=True).start()
            # engine.say(text) # Blocking call
            engine.runAndWait() # Ensure say() commands are processed
        except Exception as e:
            print(f"TTS Error: {e}")
    else:
        print(f"TTS Disabled: {text}")

# --- Configuration ---
DATA_DIR = 'data/'
MODEL_PKL = os.path.join(DATA_DIR, 'model.pkl')
VOTES_CSV = os.path.join(DATA_DIR, 'Votes.csv')
CASCADE_PATH = os.path.join('cascades', 'haarcascade_frontalface_default.xml')
BACKGROUND_IMG_PATH = os.path.join('assets', 'background.png') # Optional background
FACE_SIZE = (50, 50)
COL_NAMES = ['AADHAR', 'VOTE', 'DATE', 'TIME']
PARTIES = ["BJP", "CONGRESS", "AAP", "NOTA"]
MIN_PREDICTION_FRAMES = 5 # Require consecutive identical predictions for confidence
FACE_CONFIDENCE_THRESHOLD = 1.3
MIN_NEIGHBORS = 5

# --- Global Variables ---
video = None
knn_model = None
facedetect = None
voted_ids = set()
last_prediction = None
prediction_streak = 0
recognized_aadhar = None # Store the Aadhar number when recognized confidently

# --- Helper Functions ---
def load_model():
    """ Loads the trained KNN model. """
    global knn_model
    if not os.path.exists(MODEL_PKL):
        messagebox.showerror("Error", f"Model file not found: {MODEL_PKL}\nPlease run 2_train_model.py first.")
        return False
    try:
        with open(MODEL_PKL, 'rb') as f:
            knn_model = pickle.load(f)
        print("KNN Model loaded successfully.")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
        return False

def load_cascade():
    """ Loads the face detection cascade. """
    global facedetect
    if not os.path.exists(CASCADE_PATH):
         messagebox.showerror("Error", f"Cascade file not found: {CASCADE_PATH}")
         return False
    facedetect = cv2.CascadeClassifier(CASCADE_PATH)
    if facedetect.empty():
         messagebox.showerror("Error", f"Failed to load cascade file: {CASCADE_PATH}")
         return False
    print("Face detector loaded.")
    return True

def load_voted_ids():
    """ Loads already voted Aadhar numbers from CSV into a set for fast lookup. """
    global voted_ids
    voted_ids = set()
    if os.path.exists(VOTES_CSV):
        try:
            with open(VOTES_CSV, "r", newline='') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader, None) # Skip header
                if header: # Basic check if header matches expected
                    if header[0] != COL_NAMES[0]:
                         print("Warning: Votes.csv header mismatch. Assuming first column is Aadhar.")

                for row in reader:
                    if row: # Check if row is not empty
                        voted_ids.add(row[0]) # Assuming Aadhar is the first column
            print(f"Loaded {len(voted_ids)} previously voted IDs.")
        except Exception as e:
            print(f"Error reading {VOTES_CSV}: {e}. Starting with empty voted list.")
            voted_ids = set() # Reset on error


def record_vote(aadhar, party):
    """ Records the vote in the CSV file. """
    global voted_ids
    if aadhar in voted_ids:
        print(f"Aadhar {aadhar} has already voted (internal check).") # Should be caught earlier
        return False

    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S") # Use : instead of - for time
    vote_record = [aadhar, party, date, timestamp]

    try:
        file_exists = os.path.isfile(VOTES_CSV)
        with open(VOTES_CSV, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES) # Write header if file is new
            writer.writerow(vote_record)
        voted_ids.add(aadhar) # Update the set
        print(f"Vote recorded: {vote_record}")
        return True
    except Exception as e:
        messagebox.showerror("Error", f"Failed to write vote to {VOTES_CSV}: {e}")
        return False

# --- GUI Application Class ---
class VotingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Voting System")
        # self.root.geometry("1000x700") # Adjust size as needed
        # --- Force focus after setup ---
        self.root.update_idletasks() # Ensure window exists before forcing focus
        self.root.focus_force()
        # ------------------------------

        # --- Fonts ---
        self.header_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.status_font = tkFont.Font(family="Helvetica", size=12)
        self.button_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # --- Load Background (Optional) ---
        try:
            self.bg_image = Image.open(BACKGROUND_IMG_PATH)
            # Resize background to fit window approx (adjust size)
            # self.bg_image = self.bg_image.resize((1000, 700), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(self.bg_image)
            self.bg_label = tk.Label(root, image=self.bg_photo)
            self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except FileNotFoundError:
            print(f"Background image not found at {BACKGROUND_IMG_PATH}. Using plain background.")
        except Exception as e:
             print(f"Error loading background image: {e}")


        # --- Video Display Label ---
        self.video_label = tk.Label(root) # No background initially
        self.video_label.pack(pady=20)

        # --- Status Label ---
        self.status_label = tk.Label(root, text="Initializing...", font=self.status_font, bg="lightblue", fg="black", width=50, height=2)
        self.status_label.pack(pady=10)

        # --- Voting Buttons Frame ---
        self.button_frame = tk.Frame(root, bg=root.cget('bg')) # Match root background
        self.button_frame.pack(pady=20)

        self.vote_buttons = {}
        for i, party in enumerate(PARTIES):
            button = tk.Button(self.button_frame, text=f"Vote {party}", font=self.button_font,
                               command=lambda p=party: self.cast_vote(p),
                               width=15, height=2, state=tk.DISABLED) # Start disabled
            button.grid(row=0, column=i, padx=10, pady=5)
            self.vote_buttons[party] = button

        # --- Initialize Video and Model ---
        self.update_status("Loading resources...")
        if not load_cascade() or not load_model():
             self.update_status("Error: Failed to load resources. Check console.", error=True)
             self.root.after(5000, self.root.quit) # Quit after 5s if load fails
             return

        load_voted_ids() # Load previously voted IDs

        self.update_status("Initializing camera...")
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            self.update_status("Error: Cannot open camera.", error=True)
            messagebox.showerror("Camera Error", "Could not access the camera.")
            return

        self.update_status("Place face in front of camera.")
        self.update_frame() # Start the video loop

    def update_status(self, text, error=False):
        """ Updates the status label text and color. """
        self.status_label.config(text=text, fg="red" if error else "black", bg="lightcoral" if error else "lightblue")
        print(f"Status: {text}") # Also print to console

    def update_frame(self):
        """ Captures frame, performs detection/recognition, updates GUI. """
        global last_prediction, prediction_streak, recognized_aadhar

        ret, frame = self.video.read()
        if not ret:
            self.update_status("Error: Failed to grab frame.", error=True)
            self.root.after(1000, self.update_frame) # Try again after 1s
            return

        # Prepare frame for display
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, FACE_CONFIDENCE_THRESHOLD, MIN_NEIGHBORS)

        current_recognized_aadhar = None # Aadhar recognized in THIS frame

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, FACE_SIZE)
            flattened_img = resized_img.flatten().reshape(1, -1)

            # Predict using the loaded model
            try:
                output = knn_model.predict(flattened_img)
                aadhar = output[0]
                current_recognized_aadhar = aadhar # Store Aadhar recognized in this frame

                # --- Recognition Confidence (Streak Check) ---
                if aadhar == last_prediction:
                    prediction_streak += 1
                else:
                    last_prediction = aadhar
                    prediction_streak = 1
                    recognized_aadhar = None # Reset confirmed Aadhar if prediction changes
                    self.disable_voting_buttons() # Disable buttons if recognition unstable

                # --- Update GUI based on recognition ---
                if prediction_streak >= MIN_PREDICTION_FRAMES:
                    # Confident recognition
                    recognized_aadhar = aadhar # Store confirmed Aadhar
                    display_text = f"ID: {aadhar[-4:]}****" # Mask Aadhar partially
                    box_color = (0, 255, 0) # Green for confident

                    if aadhar in voted_ids:
                        self.update_status(f"Voter {aadhar[-4:]}**** has ALREADY VOTED.", error=True)
                        box_color = (0, 0, 255) # Red if already voted
                        speak(f"Voter {aadhar} has already voted.") # Speak full Aadhar here? maybe not
                        self.disable_voting_buttons()
                    else:
                         self.update_status(f"Voter {aadhar[-4:]}**** recognized. Please vote.")
                         self.enable_voting_buttons() # Enable buttons only when confident & not voted

                else:
                    # Recognition not yet confident
                    display_text = "Identifying..."
                    box_color = (0, 255, 255) # Yellow for identifying
                    self.disable_voting_buttons()

                # --- Draw on frame ---
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                cv2.rectangle(display_frame, (x, y - 40), (x + w, y), box_color, -1)
                cv2.putText(display_frame, display_text, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                print(f"Prediction Error: {e}")
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # Blue for error

            break # Process only the first detected face

        # If no face detected in this frame
        if len(faces) == 0:
             last_prediction = None
             prediction_streak = 0
             recognized_aadhar = None
             self.disable_voting_buttons()
             if self.status_label.cget("text") != "Initializing..." and not "already voted" in self.status_label.cget("text").lower():
                 self.update_status("Place face in front of camera.")


        # Convert frame for Tkinter display
        cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk # Keep reference
        self.video_label.config(image=imgtk)

        # Schedule next frame update
        self.root.after(20, self.update_frame) # Update roughly 50fps

    def enable_voting_buttons(self):
        for button in self.vote_buttons.values():
            button.config(state=tk.NORMAL)

    def disable_voting_buttons(self):
        for button in self.vote_buttons.values():
            button.config(state=tk.DISABLED)

    def cast_vote(self, party):
        global recognized_aadhar
        if recognized_aadhar and recognized_aadhar not in voted_ids:
            # Optional: Confirmation Dialog
            confirm = messagebox.askyesno("Confirm Vote", f"Confirm vote for {party} for Aadhar ...{recognized_aadhar[-4:]}?")
            if confirm:
                speak(f"Recording vote for {party}.")
                if record_vote(recognized_aadhar, party):
                    self.update_status(f"Vote for {party} recorded for ...{recognized_aadhar[-4:]}. Thank you!", error=False)
                    speak("Vote Recorded. Thank you for participating.")
                    self.disable_voting_buttons()
                    recognized_aadhar = None # Reset after voting
                    last_prediction = None
                    prediction_streak = 0
                    # Optional: Close window after vote?
                    # self.root.after(3000, self.root.quit)
                else:
                    self.update_status(f"Failed to record vote for ...{recognized_aadhar[-4:]}.", error=True)
                    speak("Error recording vote.")
            else:
                 self.update_status(f"Vote for {party} cancelled.")
        else:
            print("Cannot vote: No recognized Aadhar or already voted.")
            speak("Cannot vote now.") # Generic error

    def on_close(self):
        """ Release resources when window is closed. """
        print("Closing application...")
        if self.video and self.video.isOpened():
            self.video.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VotingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close) # Handle window close event
    root.mainloop()