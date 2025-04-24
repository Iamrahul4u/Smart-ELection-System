import tkinter as tk
from tkinter import ttk, font as tkFont, messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
import threading
from sklearn.neighbors import KNeighborsClassifier # Import for KNN

# --- Optional: Cross-platform Text-to-Speech ---
try:
    import pyttsx3
    engine = pyttsx3.init()
except ImportError:
    print("Warning: pyttsx3 not found. Text-to-speech disabled.")
    engine = None
except Exception as e_tts_init:
    print(f"Error initializing pyttsx3: {e_tts_init}")
    engine = None

def speak_async(text):
    """ Speaks the given text using pyttsx3 if available. Runs in a separate thread. """
    def speak_worker(text_to_speak):
        if engine:
            try:
                # Check if engine is busy, might need queuing in complex apps
                engine.say(text_to_speak)
                engine.runAndWait() # Process commands in this thread
            except Exception as e:
                print(f"TTS Error in worker: {e}")
        else:
            print(f"TTS Disabled: {text_to_speak}")

    # Start the speaking in a new daemon thread
    threading.Thread(target=speak_worker, args=(text,), daemon=True).start()


# --- Configuration (Consolidated) ---
DATA_DIR = 'data/'
FACES_PKL = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL = os.path.join(DATA_DIR, 'names.pkl')
MODEL_PKL = os.path.join(DATA_DIR, 'model.pkl')
VOTES_CSV = os.path.join(DATA_DIR, 'Votes.csv')
CASCADE_PATH = os.path.join('cascades', 'haarcascade_frontalface_default.xml') # Ensure this path is correct
FACE_SIZE = (50, 50)
FRAMES_TO_CAPTURE = 50
CAPTURE_DELAY_MS = 100 # Capture roughly every 100ms
FACE_CONFIDENCE_THRESHOLD = 1.3 # Haar cascade scaleFactor
MIN_NEIGHBORS = 5 # Haar cascade minNeighbors
N_NEIGHBORS_KNN = 5
MIN_PREDICTION_FRAMES_VOTE = 5 # Require consecutive identical predictions for confidence
RESET_DELAY_MS_VOTE = 4000 # Time in ms to show status message before reset
COL_NAMES_VOTES = ['AADHAR', 'VOTE', 'DATE', 'TIME']
PARTIES = ["BJP", "CONGRESS", "AAP", "NOTA"]

# --- Helper Functions ---
def ensure_dir_exists(path):
    """ Create directory if it doesn't exist. """
    dir_path = os.path.dirname(path)
    # Handle cases where path is just a directory or has no directory part
    if dir_path and not os.path.exists(dir_path):
         print(f"Creating directory: {dir_path}")
         os.makedirs(dir_path)
    elif not dir_path and not os.path.exists(path) and '.' not in os.path.basename(path): # If path itself is a directory
        print(f"Creating directory: {path}")
        os.makedirs(path)


def validate_aadhar(aadhar_str):
    """ Basic validation for Aadhar number (numeric, specific length). """
    return aadhar_str.isdigit() and len(aadhar_str) == 12 # Assuming 12 digits

# --- Main Application Class ---
class FaceVoteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Voting System")
        self.root.geometry("800x750") # Adjust as needed

        # --- State Variables ---
        self.current_mode = None # 'enroll', 'vote', None
        self.video_capture = None
        self.facedetect = None
        self.knn_model = None
        self.voted_ids = set()
        self.current_enroll_aadhar = ""
        self.enroll_faces_data = []
        self.is_capturing = False
        self.is_voting_active = False
        self.last_prediction_vote = None
        self.prediction_streak_vote = 0
        self.recognized_aadhar_vote = None # Store the Aadhar number when recognized confidently
        self.status_reset_timer_id = None # To store the ID of the pending reset timer

        # --- Load Resources ---
        ensure_dir_exists(DATA_DIR)
        ensure_dir_exists(os.path.join(DATA_DIR, 'placeholder.txt')) # Ensure data dir itself exists
        ensure_dir_exists(os.path.join('cascades', 'placeholder.txt')) # Ensure cascades dir exists
        self._load_cascade()
        self._load_voted_ids()
        # Model is loaded only when switching to vote mode or during training complete

        # --- Styling ---
        self.header_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = tkFont.Font(family="Helvetica", size=12)
        self.status_font = tkFont.Font(family="Helvetica", size=11)
        self.button_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        style = ttk.Style()
        style.theme_use('clam') # Use a theme that looks slightly better

        # --- Main Frame ---
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Header ---
        self.header_label = ttk.Label(self.main_frame, text="Voting System", font=self.header_font, anchor=tk.CENTER)
        self.header_label.pack(pady=10)

        # --- Mode Selection Buttons ---
        self.mode_frame = ttk.Frame(self.main_frame)
        self.mode_frame.pack(pady=10)

        self.enroll_button = ttk.Button(self.mode_frame, text="Enroll New Voter", command=self.show_enroll_mode, width=20)
        self.enroll_button.grid(row=0, column=0, padx=10)

        self.vote_button = ttk.Button(self.mode_frame, text="Start Voting Session", command=self.show_vote_mode, width=20)
        self.vote_button.grid(row=0, column=1, padx=10)

        self.train_button = ttk.Button(self.mode_frame, text="Train Recognition Model", command=self.run_training_thread, width=25)
        self.train_button.grid(row=0, column=2, padx=10)

        # --- Content Frame (parent for enroll/vote frames) ---
        self.content_frame = ttk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        self.content_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.content_frame.grid_rowconfigure(0, weight=1)    # Make row 0 expandable
        self.content_frame.grid_columnconfigure(0, weight=1) # Make column 0 expandable

        # --- Status Bar ---
        self.status_label = ttk.Label(self.main_frame, text="Welcome! Select an option.", font=self.status_font, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Initialize Mode Frames (attached to content_frame) ---
        # IMPORTANT: Pass self.content_frame as the parent
        self.enroll_frame = self._create_enroll_frame(self.content_frame)
        self.vote_frame = self._create_vote_frame(self.content_frame)

        # Hide frames initially using grid_remove
        self.enroll_frame.grid_remove()
        self.vote_frame.grid_remove()

        # --- Video Update Loop ---
        self.update_video_feed()

        # --- Set focus ---
        self.root.update_idletasks()
        self.root.lift()
        self.root.focus_force()


    # --- Resource Loading ---
    def _load_cascade(self):
        if not os.path.exists(CASCADE_PATH):
            self._update_status(f"ERROR: Cascade file not found: {CASCADE_PATH}", error=True)
            messagebox.showerror("Error", f"Cascade file not found:\n{CASCADE_PATH}")
            self.facedetect = None
            return
        self.facedetect = cv2.CascadeClassifier(CASCADE_PATH)
        if self.facedetect.empty():
            self._update_status(f"ERROR: Failed to load cascade: {CASCADE_PATH}", error=True)
            messagebox.showerror("Error", f"Failed to load cascade file:\n{CASCADE_PATH}")
            self.facedetect = None
        else:
            print("Face detector loaded.")

    def _load_model(self):
        if not os.path.exists(MODEL_PKL):
            self._update_status(f"ERROR: Model not found. Please Train Model.", error=True)
            # Don't show messagebox here, let the caller handle UI feedback
            self.knn_model = None
            return False
        try:
            # Add check for empty model file
            if os.path.getsize(MODEL_PKL) == 0:
                 print("Warning: Model file exists but is empty.")
                 self._update_status(f"Error: Model file is empty. Please re-train.", error=True)
                 self.knn_model = None
                 return False
            with open(MODEL_PKL, 'rb') as f:
                self.knn_model = pickle.load(f)
            print("KNN Model loaded successfully.")
            return True
        except EOFError: # Specific error for empty/corrupt pickle
            print(f"Error: Model file {MODEL_PKL} is empty or corrupted.")
            self._update_status(f"Error: Model file is empty/corrupt. Please re-train.", error=True)
            self.knn_model = None
            return False
        except Exception as e:
            self._update_status(f"Error loading model: {e}", error=True)
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
            self.knn_model = None
            return False

    def _load_voted_ids(self):
        self.voted_ids = set()
        if os.path.exists(VOTES_CSV):
            try:
                with open(VOTES_CSV, "r", newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    try:
                        header = next(reader, None) # Skip header safely
                        if header: # Basic check if header matches expected
                           if header[0].strip().upper() != COL_NAMES_VOTES[0]:
                               print(f"Warning: Votes.csv header ('{header[0]}') mismatch. Assuming first column is Aadhar.")
                    except StopIteration: # File is empty or only header
                         print(f"{VOTES_CSV} is empty or contains only a header.")
                         return # No IDs to load

                    # Load actual data rows
                    for row in reader:
                        if row and len(row) > 0: # Check if row is not empty and has at least one column
                            self.voted_ids.add(row[0].strip()) # Assuming Aadhar is the first column, strip whitespace

                print(f"Loaded {len(self.voted_ids)} previously voted IDs.")
            except Exception as e:
                print(f"Error reading {VOTES_CSV}: {e}. Starting with empty voted list.")
                self.voted_ids = set() # Reset on error


    # --- Mode Switching ---
    def _switch_frame(self, frame_to_show):
        """Hides all mode frames and shows the specified one using grid."""
        if self.enroll_frame:
            self.enroll_frame.grid_remove()
        if self.vote_frame:
            self.vote_frame.grid_remove()

        if frame_to_show:
            frame_to_show.grid(row=0, column=0, sticky="nsew") # Place the frame in the content_frame's grid


    def show_enroll_mode(self):
        if self.current_mode == 'enroll': return
        # Need cascade detector for enrollment
        if not self.facedetect:
            messagebox.showerror("Error", "Face detector failed to load. Cannot start enrollment.")
            return
        self._switch_frame(self.enroll_frame)
        self.current_mode = 'enroll'
        self.is_voting_active = False # Stop voting logic
        self.header_label.config(text="Enroll New Voter")
        self._update_status("Enter Aadhar number and capture face.")
        if self.enroll_aadhar_entry: self.enroll_aadhar_entry.focus() # Check exists
        self.enroll_faces_data = []
        self.is_capturing = False
        if self.enroll_capture_button: self.enroll_capture_button.config(text="Start Capture", state=tk.NORMAL) # Check exists
        if self.enroll_save_button: self.enroll_save_button.config(state=tk.DISABLED) # Check exists
        if self.enroll_progress_label: self.enroll_progress_label.config(text="") # Check exists

    def show_vote_mode(self):
        if self.current_mode == 'vote': return
        # Need cascade detector and a trained model for voting
        if not self.facedetect:
             messagebox.showerror("Error", "Face detector failed to load. Cannot start voting.")
             return
        if not self.knn_model and not self._load_model(): # Load if not loaded, check success
             messagebox.showerror("Model Error", f"Model file not found or failed to load:\n{MODEL_PKL}\nPlease use 'Train Recognition Model' first.")
             self._update_status("Cannot start voting: Model not loaded/trained.", error=True)
             return

        self._switch_frame(self.vote_frame)
        self.current_mode = 'vote'
        self.is_voting_active = True # Start voting logic
        self.header_label.config(text="Voting Session")
        self._update_status("Place face in front of camera.")
        self._reset_vote_state()

    # --- GUI Creation (Using grid layout internally) ---
    def _create_enroll_frame(self, parent):
        frame = ttk.Frame(parent)
        # Configure grid columns/rows for the frame itself
        frame.columnconfigure(0, weight=1) # Make column 0 horizontally expandable
        # Configure row weights: Give row 2 (video) the most weight to expand vertically
        frame.rowconfigure(0, weight=0) # Input row - no extra space
        frame.rowconfigure(1, weight=0) # Progress row - no extra space
        frame.rowconfigure(2, weight=1) # Video row - takes up available vertical space
        frame.rowconfigure(3, weight=0) # Button row - no extra space

        # --- Input Row (Row 0) ---
        input_frame = ttk.Frame(frame)
        # Place input_frame in grid, make it stretch horizontally
        input_frame.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        # Pack items inside this sub-frame easily left-to-right
        ttk.Label(input_frame, text="Aadhar Number (12 digits):", font=self.label_font).pack(side=tk.LEFT, padx=5)
        self.enroll_aadhar_entry = ttk.Entry(input_frame, width=15, font=self.label_font)
        self.enroll_aadhar_entry.pack(side=tk.LEFT, padx=5)

        # --- Progress Label (Row 1) ---
        self.enroll_progress_label = ttk.Label(frame, text="", font=self.status_font, anchor=tk.CENTER)
        # Place progress label in grid, make it stretch horizontally
        self.enroll_progress_label.grid(row=1, column=0, pady=5, sticky="ew")

        # --- Video Display (Row 2 - Expandable) ---
        self.enroll_video_label = ttk.Label(frame, background='gray70') # Darker grey background
        # Place video label in grid, make it expand in all directions (nsew)
        self.enroll_video_label.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        # --- Buttons (Row 3) ---
        button_frame = ttk.Frame(frame)
        # Place button_frame in grid, center it horizontally maybe? Use default anchor.
        button_frame.grid(row=3, column=0, pady=(5, 10))
        # Pack buttons inside this sub-frame easily left-to-right
        self.enroll_capture_button = ttk.Button(button_frame, text="Start Capture", command=self.toggle_capture, width=15)
        self.enroll_capture_button.pack(side=tk.LEFT, padx=10)
        self.enroll_save_button = ttk.Button(button_frame, text="Save Enrollment", command=self.save_enrollment, width=15, state=tk.DISABLED)
        self.enroll_save_button.pack(side=tk.LEFT, padx=10)

        return frame # Return the created frame

    def _create_vote_frame(self, parent):
        frame = ttk.Frame(parent)
        # Configure grid columns/rows for the frame itself
        frame.columnconfigure(0, weight=1) # Make column 0 horizontally expandable
        frame.rowconfigure(0, weight=1) # Make row 0 (video) expandable vertically
        frame.rowconfigure(1, weight=0) # Row 1 (buttons) - no extra vertical space

        # --- Video Display (Row 0 - Expandable) ---
        self.vote_video_label = ttk.Label(frame, background='gray70') # Darker grey background
        # Place video label in grid, make it expand in all directions (nsew)
        self.vote_video_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # --- Voting Buttons Frame (Row 1) ---
        self.vote_button_frame = ttk.Frame(frame)
         # Place button_frame in grid, center it horizontally maybe? Use default anchor.
        self.vote_button_frame.grid(row=1, column=0, pady=10)

        self.vote_buttons_widgets = {}
        for i, party in enumerate(PARTIES):
            # Grid buttons within the button_frame
            button = ttk.Button(self.vote_button_frame, text=f"Vote {party}",
                               command=lambda p=party: self.cast_vote(p),
                               width=15, style='Vote.TButton', state=tk.DISABLED) # Use custom style later if needed
            button.grid(row=0, column=i, padx=10, pady=5)
            self.vote_buttons_widgets[party] = button

        return frame # Return the created frame


    # --- Enrollment Logic ---
    def toggle_capture(self):
        if self.is_capturing:
            self.is_capturing = False
            if self.enroll_capture_button and self.enroll_capture_button.winfo_exists(): self.enroll_capture_button.config(text="Start Capture") # Check exists
            if len(self.enroll_faces_data) >= FRAMES_TO_CAPTURE // 2:
                if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.NORMAL) # Check exists
                self._update_status(f"Capture stopped. {len(self.enroll_faces_data)} frames collected. Ready to Save.")
            else:
                 self._update_status(f"Capture stopped. Only {len(self.enroll_faces_data)} frames collected. Not enough data.")
                 if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED) # Check exists
        else:
            aadhar = self.enroll_aadhar_entry.get()
            if not validate_aadhar(aadhar):
                messagebox.showerror("Invalid Input", "Please enter a valid 12-digit Aadhar number.")
                self._update_status("Invalid Aadhar number.", error=True)
                return

            self.current_enroll_aadhar = aadhar
            self.enroll_faces_data = []
            self.is_capturing = True
            if self.enroll_capture_button and self.enroll_capture_button.winfo_exists(): self.enroll_capture_button.config(text="Stop Capture") # Check exists
            if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED) # Check exists
            self._update_status(f"Capturing face for {aadhar[-4:]}****...")
            print(f"Starting capture for Aadhar: {aadhar}")

    def save_enrollment(self):
        if not self.current_enroll_aadhar or not self.enroll_faces_data:
            messagebox.showerror("Error", "No enrollment data to save.")
            return

        if len(self.enroll_faces_data) < FRAMES_TO_CAPTURE // 2:
             messagebox.showwarning("Warning", f"Only {len(self.enroll_faces_data)} frames captured. Data might be insufficient.")
             # Decide whether to proceed or return
             # return # Option: Don't save insufficient data

        try:
            faces_data_np = np.asarray(self.enroll_faces_data)
            # Ensure it's 2D (samples, features) - flatten correctly
            n_samples = len(self.enroll_faces_data)
            faces_data_np = faces_data_np.reshape(n_samples, -1) # Flatten features

            # Append Names
            if os.path.exists(NAMES_PKL) and os.path.getsize(NAMES_PKL) > 0:
                 with open(NAMES_PKL, 'rb') as f: names = pickle.load(f)
            else: names = []
            names.extend([self.current_enroll_aadhar] * n_samples)
            with open(NAMES_PKL, 'wb') as f: pickle.dump(names, f)

            # Append Faces
            if os.path.exists(FACES_PKL) and os.path.getsize(FACES_PKL) > 0:
                with open(FACES_PKL, 'rb') as f: faces = pickle.load(f)
                # Check if loaded data is 2D
                if faces.ndim != 2:
                     raise ValueError(f"Existing faces data has incorrect dimensions ({faces.ndim}). Expected 2. Please check or delete {FACES_PKL}.")
                if faces.shape[1] == faces_data_np.shape[1]:
                    faces = np.append(faces, faces_data_np, axis=0)
                else:
                    # Attempt to reshape if the total number of elements matches
                    # This is risky and assumes previous data was saved incorrectly
                    if faces.size == faces_data_np.shape[0] * faces_data_np.shape[1]:
                         print("Warning: Reshaping existing face data to match new data. Check data integrity.")
                         faces = faces.reshape(-1, faces_data_np.shape[1])
                         faces = np.append(faces, faces_data_np, axis=0)
                    else:
                         raise ValueError(f"Shape mismatch! Existing features {faces.shape[1]}, New features {faces_data_np.shape[1]}")
            else: faces = faces_data_np # First time saving or empty file
            with open(FACES_PKL, 'wb') as f: pickle.dump(faces, f)

            self._update_status(f"Enrollment saved for {self.current_enroll_aadhar[-4:]}****. Train Model Recommended.")
            messagebox.showinfo("Success", f"Enrollment data saved successfully for Aadhar ...{self.current_enroll_aadhar[-4:]}!\nRemember to Train the Model if this is new data.")
            # Reset fields after successful save
            self.enroll_faces_data = []
            self.current_enroll_aadhar = ""
            if self.enroll_aadhar_entry and self.enroll_aadhar_entry.winfo_exists(): self.enroll_aadhar_entry.delete(0, tk.END) # Check exists
            if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED) # Check exists
            if self.enroll_progress_label and self.enroll_progress_label.winfo_exists(): self.enroll_progress_label.config(text="") # Check exists


        except ValueError as ve:
             messagebox.showerror("Save Error", f"Data error: {ve}")
             self._update_status(f"Error saving enrollment: {ve}", error=True)
        except Exception as e:
             messagebox.showerror("Save Error", f"Failed to save enrollment data:\n{e}")
             self._update_status(f"Error saving enrollment: {e}", error=True)


    # --- Training Logic ---
    def run_training_thread(self):
        # Disable buttons during training
        if self.enroll_button and self.enroll_button.winfo_exists(): self.enroll_button.config(state=tk.DISABLED)
        if self.vote_button and self.vote_button.winfo_exists(): self.vote_button.config(state=tk.DISABLED)
        if self.train_button and self.train_button.winfo_exists(): self.train_button.config(state=tk.DISABLED)
        self._update_status("Starting model training... Please wait.")

        # Run training in a separate thread
        train_thread = threading.Thread(target=self._train_model_worker, daemon=True)
        train_thread.start()

    def _train_model_worker(self):
        """Worker function to run training in the background."""
        success = False
        message = "Training failed."
        loaded_knn = None # Store loaded model temporarily
        try:
            print("Loading data for training...")
            if not os.path.exists(FACES_PKL) or not os.path.exists(NAMES_PKL):
                message = "ERROR: Data files not found. Please Enroll faces first."
                raise FileNotFoundError(message)

            # Check if files are empty before loading pickle
            if os.path.getsize(FACES_PKL) == 0 or os.path.getsize(NAMES_PKL) == 0:
                 message = "ERROR: Data files are empty. Please Enroll faces."
                 raise ValueError(message)

            with open(FACES_PKL, 'rb') as f: faces = pickle.load(f)
            with open(NAMES_PKL, 'rb') as f: names = pickle.load(f)

            if len(faces) != len(names):
                message = f"ERROR: Data mismatch! Samples: {len(faces)}, Labels: {len(names)}"
                raise ValueError(message)
            if faces.size == 0: # Check if faces array is empty
                 message = "ERROR: No face data found to train."
                 raise ValueError(message)
            if faces.ndim != 2: # Ensure data is 2D (n_samples, n_features)
                message = f"ERROR: Face data has incorrect dimensions ({faces.ndim}). Expected 2 (samples, features). Check data saving/loading."
                raise ValueError(message)


            print(f"Training KNN model on {len(faces)} samples for {len(np.unique(names))} unique IDs...")
            # Use the imported KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_KNN, weights='distance', metric='euclidean')
            knn.fit(faces, names) # faces should already be flattened

            print("Saving trained model...")
            with open(MODEL_PKL, 'wb') as f: pickle.dump(knn, f)
            message = f"Model training complete ({len(faces)} samples) and saved successfully!"
            success = True
            loaded_knn = knn # Store the newly trained model
            print(message)

        except FileNotFoundError as e:
            print(f"Training Error: {e}")
            message = str(e) # Use the specific error message
            success = False
        except ValueError as e: # Catch specific errors like empty data or shape mismatch
            print(f"Training Data Error: {e}")
            message = str(e)
            success = False
        except Exception as e: # Catch other potential errors during training/saving
            print(f"Generic Training Error: {e}")
            message = f"An unexpected error occurred during training: {e}"
            success = False

        # Update GUI from the main thread after training finishes
        # Ensure root still exists before scheduling GUI update
        if self.root.winfo_exists():
            # Pass the newly trained model (or None if failed) to the completion handler
            self.root.after(0, self._on_training_complete, success, message, loaded_knn)

    def _on_training_complete(self, success, message, trained_model):
        """Called after training thread finishes to update GUI."""
        if success:
            messagebox.showinfo("Training Complete", message)
            # Update the app's model instance with the newly trained one
            self.knn_model = trained_model
        else:
            messagebox.showerror("Training Error", message)
        self._update_status(message, error=not success)
        # Re-enable buttons (check if they exist first)
        if self.enroll_button and self.enroll_button.winfo_exists(): self.enroll_button.config(state=tk.NORMAL)
        # Enable vote button only if model training was successful and model loaded
        if self.vote_button and self.vote_button.winfo_exists(): self.vote_button.config(state=tk.NORMAL if success and self.knn_model else tk.DISABLED)
        if self.train_button and self.train_button.winfo_exists(): self.train_button.config(state=tk.NORMAL)


    # --- Voting Logic ---
    def _reset_vote_state(self):
        self.last_prediction_vote = None
        self.prediction_streak_vote = 0
        self.recognized_aadhar_vote = None
        self._disable_vote_buttons()
        self._cancel_status_reset_timer() # Ensure timer is cancelled on state reset

    def _enable_vote_buttons(self):
        # Check if frame exists before trying to access children
        if hasattr(self, 'vote_buttons_widgets') and self.vote_buttons_widgets:
             for button in self.vote_buttons_widgets.values():
                 # Ensure GUI updates happen in the main thread and widget exists
                 if button and button.winfo_exists():
                      self.root.after(0, lambda b=button: b.config(state=tk.NORMAL))

    def _disable_vote_buttons(self):
        # Check if frame exists before trying to access children
        if hasattr(self, 'vote_buttons_widgets') and self.vote_buttons_widgets:
            for button in self.vote_buttons_widgets.values():
                # Ensure GUI updates happen in the main thread and widget exists
                if button and button.winfo_exists():
                    self.root.after(0, lambda b=button: b.config(state=tk.DISABLED))

    def _reset_status_after_delay(self):
        """Resets the status message after a delay."""
        print("Resetting voting status message.")
        # Check current mode before resetting status
        if self.current_mode == 'vote':
            self._update_status("Place face in front of camera.")
        self.status_reset_timer_id = None # Clear the timer ID

    def _cancel_status_reset_timer(self):
        """Cancels any pending status reset timer."""
        if self.status_reset_timer_id is not None:
            # Check if root still exists before cancelling
            if self.root.winfo_exists():
                 try:
                      self.root.after_cancel(self.status_reset_timer_id)
                      print("Cancelled pending status reset.")
                 except tk.TclError:
                      print("Timer already cancelled or window destroyed.") # Handle race condition
            self.status_reset_timer_id = None


    def cast_vote(self, party):
        current_aadhar = self.recognized_aadhar_vote # Use the confirmed voter ID

        if current_aadhar and current_aadhar not in self.voted_ids:
            confirm = messagebox.askyesno("Confirm Vote", f"Confirm vote for {party} for Aadhar ...{current_aadhar[-4:]}?")
            if confirm:
                speak_async(f"Recording vote for {party}.")
                if self._record_vote_file(current_aadhar, party): # Use helper
                    self._update_status(f"Vote for {party} recorded for ...{current_aadhar[-4:]}. Thank you!")
                    speak_async("Vote Recorded. Thank you.")
                    self._reset_vote_state() # Reset state after vote
                    # Schedule status reset after thank you message
                    self._cancel_status_reset_timer() # Cancel just in case
                    # Check if root still exists before scheduling
                    if self.root.winfo_exists():
                         self.status_reset_timer_id = self.root.after(RESET_DELAY_MS_VOTE, self._reset_status_after_delay)

                else:
                    # _record_vote_file already shows messagebox and updates status
                    speak_async("Error recording vote.")
            else:
                 self._update_status(f"Vote for {party} cancelled.")
        else:
            # Handle edge cases or provide feedback
            if current_aadhar and current_aadhar in self.voted_ids:
                 msg = f"Cannot vote: ...{current_aadhar[-4:]} already voted."
                 self._update_status(msg, error=True)
                 speak_async("Already voted.")
            else:
                 msg = "Cannot vote: No voter recognized."
                 self._update_status(msg, error=True)
                 speak_async("Cannot vote now.")
            print(msg)
            self._disable_vote_buttons() # Ensure buttons are off

    def _record_vote_file(self, aadhar, party):
        """ Internal helper to write vote to CSV. """
        if aadhar in self.voted_ids: return False # Double check
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        vote_record = [aadhar, party, date, timestamp]
        try:
            ensure_dir_exists(VOTES_CSV) # Ensure directory exists before opening
            file_exists = os.path.isfile(VOTES_CSV)
            is_empty = not file_exists or os.path.getsize(VOTES_CSV) == 0
            with open(VOTES_CSV, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if is_empty: writer.writerow(COL_NAMES_VOTES) # Write header if file is new or empty
                writer.writerow(vote_record)
            self.voted_ids.add(aadhar) # Update the set
            print(f"Vote recorded: {vote_record}")
            return True
        except PermissionError:
             err_msg = f"Permission denied writing to {VOTES_CSV}. Check file permissions."
             messagebox.showerror("File Error", err_msg)
             self._update_status(err_msg, error=True)
             print(err_msg)
             return False
        except Exception as e:
            # Show error in GUI and console
            err_msg = f"Failed to write vote to {VOTES_CSV}: {e}"
            messagebox.showerror("File Error", err_msg)
            self._update_status(f"Error writing vote file: {e}", error=True)
            print(err_msg)
            return False


    # --- Video Feed Update (Handles both modes) ---
    def update_video_feed(self):
        # Determine the active video label based on current mode
        active_video_label = None
        # Check widgets exist before trying to use them
        if self.current_mode == 'enroll' and hasattr(self, 'enroll_video_label') and self.enroll_video_label.winfo_exists():
            active_video_label = self.enroll_video_label
        elif self.current_mode == 'vote' and hasattr(self, 'vote_video_label') and self.vote_video_label.winfo_exists():
            active_video_label = self.vote_video_label

        # Initialize camera if needed
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture or not self.video_capture.isOpened():
                 # Display error message on the active GUI label if camera fails
                 if active_video_label: # Check if label is determined
                      placeholder = Image.new('RGB', (320, 240), color = 'grey') # Smaller placeholder
                      imgtk = ImageTk.PhotoImage(image=placeholder)
                      active_video_label.imgtk = imgtk # Keep ref
                      active_video_label.config(image=imgtk, text="Camera Error", compound=tk.CENTER)
                 self._update_status("ERROR: Cannot access camera.", error=True)
                 # Check root exists before scheduling retry
                 if self.root.winfo_exists(): self.root.after(1000, self.update_video_feed)
                 return # Stop this update cycle

        # Grab frame
        ret, frame = self.video_capture.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            # Check root exists before scheduling retry
            if self.root.winfo_exists(): self.root.after(50, self.update_video_feed) # Try again quickly
            return # Stop this update cycle

        display_frame = frame.copy()
        processed_face_this_frame = False # Flag to track if a face was processed in this frame

        # Only process if cascade is loaded
        if self.facedetect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, FACE_CONFIDENCE_THRESHOLD, MIN_NEIGHBORS)

            if len(faces) > 0:
                 processed_face_this_frame = True
                 x, y, w, h = faces[0] # Process only the first detected face

                 # --- Enrollment Mode Logic ---
                 if self.current_mode == 'enroll' and self.is_capturing:
                     crop_img = frame[y:y+h, x:x+w]
                     resized_img = cv2.resize(crop_img, FACE_SIZE)
                     if len(self.enroll_faces_data) < FRAMES_TO_CAPTURE:
                         self.enroll_faces_data.append(resized_img)
                         # Check if label exists before configuring
                         if hasattr(self, 'enroll_progress_label') and self.enroll_progress_label.winfo_exists():
                             progress_text = f"Captured: {len(self.enroll_faces_data)}/{FRAMES_TO_CAPTURE}"
                             self.enroll_progress_label.config(text=progress_text)
                         # Auto-stop capture
                         if len(self.enroll_faces_data) == FRAMES_TO_CAPTURE:
                              self.toggle_capture() # Calls update_status internally
                              speak_async("Face capture complete. Please save.")
                     # Draw capture box
                     cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                 # --- Voting Mode Logic ---
                 elif self.current_mode == 'vote' and self.is_voting_active and self.knn_model:
                      crop_img = frame[y:y+h, x:x+w]
                      resized_img = cv2.resize(crop_img, FACE_SIZE)
                      flattened_img = resized_img.flatten().reshape(1, -1)
                      try:
                           output = self.knn_model.predict(flattened_img)
                           aadhar = output[0]
                           # Confidence streak check
                           if aadhar == self.last_prediction_vote: self.prediction_streak_vote += 1
                           else:
                               self._cancel_status_reset_timer()
                               self.last_prediction_vote = aadhar
                               self.prediction_streak_vote = 1
                               self.recognized_aadhar_vote = None
                               self._disable_vote_buttons()

                           # Update GUI based on recognition
                           if self.prediction_streak_vote >= MIN_PREDICTION_FRAMES_VOTE:
                               self.recognized_aadhar_vote = aadhar # Confirm recognition
                               display_text = f"ID: {aadhar[-4:]}****"
                               box_color = (0, 255, 0) # Green for recognized

                               if aadhar in self.voted_ids:
                                   status_msg = f"Voter {aadhar[-4:]}**** ALREADY VOTED."
                                   self._update_status(status_msg, error=True)
                                   box_color = (0, 0, 255) # Red for voted
                                   self._disable_vote_buttons()
                                   # Start reset timer only once
                                   if self.status_reset_timer_id is None:
                                       speak_async(f"Voter {aadhar} has already voted.")
                                       if self.root.winfo_exists(): # Check root before scheduling
                                            self.status_reset_timer_id = self.root.after(RESET_DELAY_MS_VOTE, self._reset_status_after_delay)
                               else: # Valid voter recognized
                                   self._cancel_status_reset_timer()
                                   status_msg = f"Voter {aadhar[-4:]}**** recognized. Please Vote."
                                   self._update_status(status_msg)
                                   self._enable_vote_buttons()
                           else: # Identifying
                                self._cancel_status_reset_timer()
                                display_text = "Identifying..."
                                box_color = (0, 255, 255) # Yellow
                                self._disable_vote_buttons()
                                # Update status only if not showing voted message
                                if self.status_reset_timer_id is None:
                                    self._update_status("Identifying face...")

                           # Draw rectangle and text
                           cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                           text_bg_y = max(0, y - 30) # Smaller text area
                           cv2.rectangle(display_frame, (x, text_bg_y), (x + w, y), box_color, -1)
                           cv2.putText(display_frame, display_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Smaller text

                      except Exception as e:
                           print(f"Vote Prediction Error: {e}")
                           cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # Blue for error
                 else:
                      # Face detected, but not in active enroll/vote mode
                      cv2.rectangle(display_frame, (x, y), (x+w, y+h), (200, 200, 200), 1)


        # --- Handle No Face Detected or Not in Active Mode ---
        if self.current_mode == 'vote' and not processed_face_this_frame:
             self._cancel_status_reset_timer()
             # Reset recognition state only if no face is actively detected
             self.last_prediction_vote = None
             self.prediction_streak_vote = 0
             self.recognized_aadhar_vote = None
             self._disable_vote_buttons()
             # Update status only if not showing voted/thank you message
             if self.status_reset_timer_id is None:
                 self._update_status("Place face in front of camera.")


        # --- Update Tkinter Image Label ---
        # Only update if the corresponding label exists and is visible
        if active_video_label and active_video_label.winfo_exists():
            try:
                # 1. Convert the *original sized* display_frame to PIL Image
                cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
                img_pil = Image.fromarray(cv2image)
                img_w, img_h = img_pil.size

                # 2. Get the current size of the Tkinter label widget
                label_w = active_video_label.winfo_width()
                label_h = active_video_label.winfo_height()

                # Avoid division by zero and handle initial size (often 1x1)
                if label_w <= 1 or label_h <= 1 or img_w == 0 or img_h == 0:
                    # Don't resize if label size is invalid
                    img_pil_resized = img_pil # Use original
                else:
                    # 3. Calculate scaling factor preserving aspect ratio
                    scale = min(label_w / img_w, label_h / img_h)

                    # 4. Calculate target dimensions
                    target_w = int(img_w * scale)
                    target_h = int(img_h * scale)

                    # 5. Resize the PIL Image (only if target size is valid)
                    if target_w > 0 and target_h > 0:
                         img_pil_resized = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    else:
                         img_pil_resized = img_pil # Fallback to original if calculation fails

                # 6. Create PhotoImage from the resized PIL image
                imgtk = ImageTk.PhotoImage(image=img_pil_resized)

                active_video_label.imgtk = imgtk # Keep reference
                active_video_label.config(image=imgtk) # Update label

            except Exception as e_img:
                print(f"Error updating GUI image: {e_img}")
        # --- End Update Tkinter Image Label ---


        # Schedule next update only if the window hasn't been destroyed
        if self.root.winfo_exists():
            # Check if video capture object still exists and is open
            is_video_open = self.video_capture and self.video_capture.isOpened()
            if is_video_open:
                self.root.after(30, self.update_video_feed) # Adjusted update rate slightly
            else:
                # If camera closed unexpectedly, stop trying to update feed
                print("Video source closed unexpectedly. Stopping feed update.")
                self._update_status("ERROR: Camera disconnected.", error=True)


    # --- Status Update Helper ---
    def _update_status(self, text, error=False):
        # Ensure GUI updates run in the main thread and widget exists
        if hasattr(self, 'status_label') and self.status_label.winfo_exists():
             self.root.after(0, lambda: self.status_label.config(text=text,
                                                             foreground="red" if error else "black",
                                                             background="lightcoral" if error else "SystemButtonFace")) # Use default bg
        print(f"Status: {text}")


    # --- Cleanup ---
    def on_close(self):
        print("Closing application...")
        self._cancel_status_reset_timer() # Cancel any pending timer
        self.is_capturing = False
        self.is_voting_active = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        # Stop TTS engine if it was running
        if engine:
            try:
                # Attempt to stop engine cleanly
                engine.stop()
                print("TTS engine stopped.")
            except Exception as e_tts_stop:
                print(f"Error stopping TTS engine: {e_tts_stop}")
        # Ensure the root window is destroyed properly
        if self.root:
            # Use try-except as root might already be destroyed in some cases
            try:
                self.root.destroy()
                print("Tkinter window destroyed.")
            except tk.TclError:
                 print("Tkinter window likely already destroyed.")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVoteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close) # Handle window close event
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Closing application.")
        app.on_close()