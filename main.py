# main_app.py (with check for existing Aadhar during enrollment)

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
from sklearn.neighbors import KNeighborsClassifier

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
    def speak_worker(text_to_speak):
        if engine:
            try:
                engine.say(text_to_speak)
                engine.runAndWait()
            except Exception as e: print(f"TTS Error: {e}")
        else: print(f"TTS Disabled: {text_to_speak}")
    threading.Thread(target=speak_worker, args=(text,), daemon=True).start()

# --- Configuration (Consolidated) ---
DATA_DIR = 'data/'
FACES_PKL = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL = os.path.join(DATA_DIR, 'names.pkl')
MODEL_PKL = os.path.join(DATA_DIR, 'model.pkl')
VOTES_CSV = os.path.join(DATA_DIR, 'Votes.csv')
CASCADE_PATH = os.path.join('cascades', 'haarcascade_frontalface_default.xml')
FACE_SIZE = (50, 50)
FRAMES_TO_CAPTURE = 50
CAPTURE_DELAY_MS = 100
FACE_CONFIDENCE_THRESHOLD = 1.3
MIN_NEIGHBORS = 5
N_NEIGHBORS_KNN = 5
MIN_PREDICTION_FRAMES_VOTE = 5
RESET_DELAY_MS_VOTE = 4000
COL_NAMES_VOTES = ['AADHAR', 'VOTE', 'DATE', 'TIME']
PARTIES = ["BJP", "CONGRESS", "AAP", "NOTA"]

# --- Helper Functions ---
def ensure_dir_exists(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
         print(f"Creating directory: {dir_path}")
         os.makedirs(dir_path)
    elif not dir_path and not os.path.exists(path) and '.' not in os.path.basename(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)

def validate_aadhar(aadhar_str):
    return aadhar_str.isdigit() and len(aadhar_str) == 12

# --- Main Application Class ---
class FaceVoteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Voting System")
        self.root.geometry("800x750")

        self.current_mode = None
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
        self.recognized_aadhar_vote = None
        self.status_reset_timer_id = None

        ensure_dir_exists(DATA_DIR)
        ensure_dir_exists(os.path.join(DATA_DIR, 'placeholder.txt'))
        ensure_dir_exists(os.path.join('cascades', 'placeholder.txt'))
        self._load_cascade()
        self._load_voted_ids()

        self.header_font = tkFont.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = tkFont.Font(family="Helvetica", size=12)
        self.status_font = tkFont.Font(family="Helvetica", size=11)
        self.button_font = tkFont.Font(family="Helvetica", size=12, weight="bold")
        style = ttk.Style()
        style.theme_use('clam')

        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.header_label = ttk.Label(self.main_frame, text="Voting System", font=self.header_font, anchor=tk.CENTER)
        self.header_label.pack(pady=10)

        self.mode_frame = ttk.Frame(self.main_frame)
        self.mode_frame.pack(pady=10)

        self.enroll_button = ttk.Button(self.mode_frame, text="Enroll New Voter", command=self.show_enroll_mode, width=20)
        self.enroll_button.grid(row=0, column=0, padx=10)

        self.vote_button = ttk.Button(self.mode_frame, text="Start Voting Session", command=self.show_vote_mode, width=20)
        self.vote_button.grid(row=0, column=1, padx=10)

        self.train_button = ttk.Button(self.mode_frame, text="Train Recognition Model", command=self.run_training_thread, width=25)
        self.train_button.grid(row=0, column=2, padx=10)

        self.content_frame = ttk.Frame(self.main_frame, relief=tk.GROOVE, borderwidth=2)
        self.content_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ttk.Label(self.main_frame, text="Welcome! Select an option.", font=self.status_font, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.enroll_frame = self._create_enroll_frame(self.content_frame)
        self.vote_frame = self._create_vote_frame(self.content_frame)

        self.enroll_frame.grid_remove()
        self.vote_frame.grid_remove()

        self.update_video_feed()

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
            self.knn_model = None
            return False
        try:
            if os.path.getsize(MODEL_PKL) == 0:
                 print("Warning: Model file exists but is empty.")
                 self._update_status(f"Error: Model file is empty. Please re-train.", error=True)
                 self.knn_model = None
                 return False
            with open(MODEL_PKL, 'rb') as f:
                self.knn_model = pickle.load(f)
            print("KNN Model loaded successfully.")
            return True
        except EOFError:
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
                        header = next(reader, None)
                        if header:
                           if header[0].strip().upper() != COL_NAMES_VOTES[0]:
                               print(f"Warning: Votes.csv header ('{header[0]}') mismatch. Assuming first column is Aadhar.")
                    except StopIteration:
                         print(f"{VOTES_CSV} is empty or contains only a header.")
                         return

                    for row in reader:
                        if row and len(row) > 0:
                            self.voted_ids.add(row[0].strip())

                print(f"Loaded {len(self.voted_ids)} previously voted IDs.")
            except Exception as e:
                print(f"Error reading {VOTES_CSV}: {e}. Starting with empty voted list.")
                self.voted_ids = set()

    def _check_aadhar_exists(self, aadhar_to_check):
        """Checks if an Aadhar number exists in the names.pkl file."""
        if not os.path.exists(NAMES_PKL):
            return False # File doesn't exist, so aadhar doesn't exist yet
        try:
            if os.path.getsize(NAMES_PKL) > 0:
                with open(NAMES_PKL, 'rb') as f:
                    loaded_names = pickle.load(f)
                    # Check if the aadhar number is present in the list
                    return aadhar_to_check in loaded_names
            else:
                print(f"Warning: {NAMES_PKL} exists but is empty during check.")
                return False # Treat empty file as aadhar not existing
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: Could not read {NAMES_PKL} for existing check (maybe corrupted?): {e}")
            # Decide: Treat as error or allow enrollment? Let's allow but warn.
            messagebox.showwarning("Data Check Warning", f"Could not verify existing enrollments due to data file issue:\n{e}\n\nAllowing enrollment, but check data files if duplicates occur.")
            return False # Allow enrollment if we cannot check
        except Exception as e:
             print(f"Unexpected error checking existing names: {e}")
             messagebox.showwarning("Data Check Warning", f"Could not verify existing enrollments due to unexpected error:\n{e}\n\nAllowing enrollment.")
             return False # Allow enrollment if we cannot check

    # --- Mode Switching ---
    def _switch_frame(self, frame_to_show):
        if self.enroll_frame: self.enroll_frame.grid_remove()
        if self.vote_frame: self.vote_frame.grid_remove()
        if frame_to_show: frame_to_show.grid(row=0, column=0, sticky="nsew")

    def show_enroll_mode(self):
        if self.current_mode == 'enroll': return
        if not self.facedetect:
            messagebox.showerror("Error", "Face detector failed to load. Cannot start enrollment.")
            return
        self._switch_frame(self.enroll_frame)
        self.current_mode = 'enroll'
        self.is_voting_active = False
        self.header_label.config(text="Enroll New Voter")
        self._update_status("Enter Aadhar number and capture face.")
        if self.enroll_aadhar_entry: self.enroll_aadhar_entry.focus()
        self.enroll_faces_data = []
        self.is_capturing = False
        if self.enroll_capture_button: self.enroll_capture_button.config(text="Start Capture", state=tk.NORMAL)
        if self.enroll_save_button: self.enroll_save_button.config(state=tk.DISABLED)
        if self.enroll_progress_label: self.enroll_progress_label.config(text="")

    def show_vote_mode(self):
        if self.current_mode == 'vote': return
        if not self.facedetect:
             messagebox.showerror("Error", "Face detector failed to load. Cannot start voting.")
             return
        if not self.knn_model and not self._load_model():
             messagebox.showerror("Model Error", f"Model file not found or failed to load:\n{MODEL_PKL}\nPlease use 'Train Recognition Model' first.")
             self._update_status("Cannot start voting: Model not loaded/trained.", error=True)
             return
        self._switch_frame(self.vote_frame)
        self.current_mode = 'vote'
        self.is_voting_active = True
        self.header_label.config(text="Voting Session")
        self._update_status("Place face in front of camera.")
        self._reset_vote_state()

    # --- GUI Creation (Using grid layout internally) ---
    def _create_enroll_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=0)
        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=1)
        frame.rowconfigure(3, weight=0)

        input_frame = ttk.Frame(frame)
        input_frame.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        ttk.Label(input_frame, text="Aadhar Number (12 digits):", font=self.label_font).pack(side=tk.LEFT, padx=5)
        self.enroll_aadhar_entry = ttk.Entry(input_frame, width=15, font=self.label_font)
        self.enroll_aadhar_entry.pack(side=tk.LEFT, padx=5)

        self.enroll_progress_label = ttk.Label(frame, text="", font=self.status_font, anchor=tk.CENTER)
        self.enroll_progress_label.grid(row=1, column=0, pady=5, sticky="ew")

        self.enroll_video_label = ttk.Label(frame, background='gray70')
        self.enroll_video_label.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, pady=(5, 10))
        self.enroll_capture_button = ttk.Button(button_frame, text="Start Capture", command=self.toggle_capture, width=15)
        self.enroll_capture_button.pack(side=tk.LEFT, padx=10)
        self.enroll_save_button = ttk.Button(button_frame, text="Save Enrollment", command=self.save_enrollment, width=15, state=tk.DISABLED)
        self.enroll_save_button.pack(side=tk.LEFT, padx=10)
        return frame

    def _create_vote_frame(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=0)

        self.vote_video_label = ttk.Label(frame, background='gray70')
        self.vote_video_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.vote_button_frame = ttk.Frame(frame)
        self.vote_button_frame.grid(row=1, column=0, pady=10)

        self.vote_buttons_widgets = {}
        for i, party in enumerate(PARTIES):
            button = ttk.Button(self.vote_button_frame, text=f"Vote {party}",
                               command=lambda p=party: self.cast_vote(p),
                               width=15, style='Vote.TButton', state=tk.DISABLED)
            button.grid(row=0, column=i, padx=10, pady=5)
            self.vote_buttons_widgets[party] = button
        return frame


    # --- Enrollment Logic ---
    def toggle_capture(self):
        if self.is_capturing:
            # Stop capture logic (unchanged)
            self.is_capturing = False
            if self.enroll_capture_button and self.enroll_capture_button.winfo_exists(): self.enroll_capture_button.config(text="Start Capture")
            if len(self.enroll_faces_data) >= FRAMES_TO_CAPTURE // 2:
                if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.NORMAL)
                self._update_status(f"Capture stopped. {len(self.enroll_faces_data)} frames collected. Ready to Save.")
            else:
                 self._update_status(f"Capture stopped. Only {len(self.enroll_faces_data)} frames collected. Not enough data.")
                 if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED)
        else:
            # --- Start Capture Logic ---
            aadhar = self.enroll_aadhar_entry.get()
            if not validate_aadhar(aadhar):
                messagebox.showerror("Invalid Input", "Please enter a valid 12-digit Aadhar number.")
                self._update_status("Invalid Aadhar number.", error=True)
                return

            # *** NEW: Check if Aadhar already exists ***
            if self._check_aadhar_exists(aadhar):
                messagebox.showwarning("Enrollment Exists", f"Aadhar number ...{aadhar[-4:]} is already enrolled.\nCannot add duplicate enrollment.")
                self._update_status(f"Aadhar ...{aadhar[-4:]} already exists. Cannot enroll again.", error=True)
                return # Stop the process here
            # ******************************************

            # If check passes, proceed with capture
            self.current_enroll_aadhar = aadhar
            self.enroll_faces_data = []
            self.is_capturing = True
            if self.enroll_capture_button and self.enroll_capture_button.winfo_exists(): self.enroll_capture_button.config(text="Stop Capture")
            if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED)
            if self.enroll_progress_label and self.enroll_progress_label.winfo_exists(): self.enroll_progress_label.config(text="") # Clear progress
            self._update_status(f"Capturing face for {aadhar[-4:]}****...")
            print(f"Starting capture for Aadhar: {aadhar}")
            # --- End Start Capture Logic ---


    def save_enrollment(self):
        # ... (save_enrollment logic - unchanged from previous version) ...
        if not self.current_enroll_aadhar or not self.enroll_faces_data:
            messagebox.showerror("Error", "No enrollment data to save.")
            return

        if len(self.enroll_faces_data) < FRAMES_TO_CAPTURE // 2:
             messagebox.showwarning("Warning", f"Only {len(self.enroll_faces_data)} frames captured. Data might be insufficient.")
             # Decide whether to proceed or return
             # return # Option: Don't save insufficient data

        try:
            faces_data_np = np.asarray(self.enroll_faces_data)
            n_samples = len(self.enroll_faces_data)
            faces_data_np = faces_data_np.reshape(n_samples, -1)

            if os.path.exists(NAMES_PKL) and os.path.getsize(NAMES_PKL) > 0:
                 with open(NAMES_PKL, 'rb') as f: names = pickle.load(f)
            else: names = []
            names.extend([self.current_enroll_aadhar] * n_samples)
            with open(NAMES_PKL, 'wb') as f: pickle.dump(names, f)

            if os.path.exists(FACES_PKL) and os.path.getsize(FACES_PKL) > 0:
                with open(FACES_PKL, 'rb') as f: faces = pickle.load(f)
                if faces.ndim != 2:
                     raise ValueError(f"Existing faces data has incorrect dimensions ({faces.ndim}). Expected 2. Please check or delete {FACES_PKL}.")
                if faces.shape[1] == faces_data_np.shape[1]:
                    faces = np.append(faces, faces_data_np, axis=0)
                else:
                     raise ValueError(f"Shape mismatch! Existing features {faces.shape[1]}, New features {faces_data_np.shape[1]}")
            else: faces = faces_data_np
            with open(FACES_PKL, 'wb') as f: pickle.dump(faces, f)

            self._update_status(f"Enrollment saved for {self.current_enroll_aadhar[-4:]}****. Train Model Recommended.")
            messagebox.showinfo("Success", f"Enrollment data saved successfully for Aadhar ...{self.current_enroll_aadhar[-4:]}!\nRemember to Train the Model if this is new data.")

            self.enroll_faces_data = []
            self.current_enroll_aadhar = ""
            if self.enroll_aadhar_entry and self.enroll_aadhar_entry.winfo_exists(): self.enroll_aadhar_entry.delete(0, tk.END)
            if self.enroll_save_button and self.enroll_save_button.winfo_exists(): self.enroll_save_button.config(state=tk.DISABLED)
            if self.enroll_progress_label and self.enroll_progress_label.winfo_exists(): self.enroll_progress_label.config(text="")


        except ValueError as ve:
             messagebox.showerror("Save Error", f"Data error: {ve}")
             self._update_status(f"Error saving enrollment: {ve}", error=True)
        except Exception as e:
             messagebox.showerror("Save Error", f"Failed to save enrollment data:\n{e}")
             self._update_status(f"Error saving enrollment: {e}", error=True)


    # --- Training Logic ---
    def run_training_thread(self):
        # ... (unchanged from previous version) ...
        if self.enroll_button and self.enroll_button.winfo_exists(): self.enroll_button.config(state=tk.DISABLED)
        if self.vote_button and self.vote_button.winfo_exists(): self.vote_button.config(state=tk.DISABLED)
        if self.train_button and self.train_button.winfo_exists(): self.train_button.config(state=tk.DISABLED)
        self._update_status("Starting model training... Please wait.")
        train_thread = threading.Thread(target=self._train_model_worker, daemon=True)
        train_thread.start()

    def _train_model_worker(self):
        # ... (unchanged from previous version) ...
        success = False
        message = "Training failed."
        loaded_knn = None
        try:
            print("Loading data for training...")
            if not os.path.exists(FACES_PKL) or not os.path.exists(NAMES_PKL):
                message = "ERROR: Data files not found. Please Enroll faces first."
                raise FileNotFoundError(message)
            if os.path.getsize(FACES_PKL) == 0 or os.path.getsize(NAMES_PKL) == 0:
                 message = "ERROR: Data files are empty. Please Enroll faces."
                 raise ValueError(message)

            with open(FACES_PKL, 'rb') as f: faces = pickle.load(f)
            with open(NAMES_PKL, 'rb') as f: names = pickle.load(f)

            if len(faces) != len(names):
                message = f"ERROR: Data mismatch! Samples: {len(faces)}, Labels: {len(names)}"
                raise ValueError(message)
            if faces.size == 0:
                 message = "ERROR: No face data found to train."
                 raise ValueError(message)
            if faces.ndim != 2:
                message = f"ERROR: Face data has incorrect dimensions ({faces.ndim}). Expected 2 (samples, features). Check data saving/loading."
                raise ValueError(message)

            print(f"Training KNN model on {len(faces)} samples for {len(np.unique(names))} unique IDs...")
            knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_KNN, weights='distance', metric='euclidean')
            knn.fit(faces, names)

            print("Saving trained model...")
            with open(MODEL_PKL, 'wb') as f: pickle.dump(knn, f)
            message = f"Model training complete ({len(faces)} samples) and saved successfully!"
            success = True
            loaded_knn = knn
            print(message)

        except FileNotFoundError as e: message = str(e); success = False
        except ValueError as e: message = str(e); success = False
        except Exception as e: message = f"An unexpected error during training: {e}"; success = False
        finally: print(f"Training Worker Finished: Success={success}, Message={message}")


        if self.root.winfo_exists():
            self.root.after(0, self._on_training_complete, success, message, loaded_knn)

    def _on_training_complete(self, success, message, trained_model):
        # ... (unchanged from previous version) ...
        if success:
            messagebox.showinfo("Training Complete", message)
            self.knn_model = trained_model
        else:
            messagebox.showerror("Training Error", message)
        self._update_status(message, error=not success)
        if self.enroll_button and self.enroll_button.winfo_exists(): self.enroll_button.config(state=tk.NORMAL)
        if self.vote_button and self.vote_button.winfo_exists(): self.vote_button.config(state=tk.NORMAL if success and self.knn_model else tk.DISABLED)
        if self.train_button and self.train_button.winfo_exists(): self.train_button.config(state=tk.NORMAL)


    # --- Voting Logic ---
    def _reset_vote_state(self):
        # ... (unchanged from previous version) ...
        self.last_prediction_vote = None
        self.prediction_streak_vote = 0
        self.recognized_aadhar_vote = None
        self._disable_vote_buttons()
        self._cancel_status_reset_timer()

    def _enable_vote_buttons(self):
        # ... (unchanged from previous version) ...
        if hasattr(self, 'vote_buttons_widgets') and self.vote_buttons_widgets:
             for button in self.vote_buttons_widgets.values():
                 if button and button.winfo_exists():
                      self.root.after(0, lambda b=button: b.config(state=tk.NORMAL))

    def _disable_vote_buttons(self):
        # ... (unchanged from previous version) ...
        if hasattr(self, 'vote_buttons_widgets') and self.vote_buttons_widgets:
            for button in self.vote_buttons_widgets.values():
                if button and button.winfo_exists():
                    self.root.after(0, lambda b=button: b.config(state=tk.DISABLED))

    def _reset_status_after_delay(self):
        # ... (unchanged from previous version) ...
        print("Resetting voting status message.")
        if self.current_mode == 'vote':
            self._update_status("Place face in front of camera.")
        self.status_reset_timer_id = None

    def _cancel_status_reset_timer(self):
        # ... (unchanged from previous version) ...
        if self.status_reset_timer_id is not None:
            if self.root.winfo_exists():
                 try:
                      self.root.after_cancel(self.status_reset_timer_id)
                      print("Cancelled pending status reset.")
                 except tk.TclError:
                      print("Timer already cancelled or window destroyed.")
            self.status_reset_timer_id = None


    def cast_vote(self, party):
        # ... (unchanged from previous version) ...
        current_aadhar = self.recognized_aadhar_vote
        if current_aadhar and current_aadhar not in self.voted_ids:
            confirm = messagebox.askyesno("Confirm Vote", f"Confirm vote for {party} for Aadhar ...{current_aadhar[-4:]}?")
            if confirm:
                speak_async(f"Recording vote for {party}.")
                if self._record_vote_file(current_aadhar, party):
                    self._update_status(f"Vote for {party} recorded for ...{current_aadhar[-4:]}. Thank you!")
                    speak_async("Vote Recorded. Thank you.")
                    self._reset_vote_state()
                    self._cancel_status_reset_timer()
                    if self.root.winfo_exists():
                         self.status_reset_timer_id = self.root.after(RESET_DELAY_MS_VOTE, self._reset_status_after_delay)
                else:
                    speak_async("Error recording vote.")
            else:
                 self._update_status(f"Vote for {party} cancelled.")
        else:
            if current_aadhar and current_aadhar in self.voted_ids:
                 msg = f"Cannot vote: ...{current_aadhar[-4:]} already voted."
                 self._update_status(msg, error=True)
                 speak_async("Already voted.")
            else:
                 msg = "Cannot vote: No voter recognized."
                 self._update_status(msg, error=True)
                 speak_async("Cannot vote now.")
            print(msg)
            self._disable_vote_buttons()

    def _record_vote_file(self, aadhar, party):
        # ... (unchanged from previous version) ...
        if aadhar in self.voted_ids: return False
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        vote_record = [aadhar, party, date, timestamp]
        try:
            ensure_dir_exists(VOTES_CSV)
            file_exists = os.path.isfile(VOTES_CSV)
            is_empty = not file_exists or os.path.getsize(VOTES_CSV) == 0
            with open(VOTES_CSV, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if is_empty: writer.writerow(COL_NAMES_VOTES)
                writer.writerow(vote_record)
            self.voted_ids.add(aadhar)
            print(f"Vote recorded: {vote_record}")
            return True
        except PermissionError:
             err_msg = f"Permission denied writing to {VOTES_CSV}. Check file permissions."
             messagebox.showerror("File Error", err_msg)
             self._update_status(err_msg, error=True)
             print(err_msg)
             return False
        except Exception as e:
            err_msg = f"Failed to write vote to {VOTES_CSV}: {e}"
            messagebox.showerror("File Error", err_msg)
            self._update_status(f"Error writing vote file: {e}", error=True)
            print(err_msg)
            return False


    # --- Video Feed Update (Handles both modes + dynamic resize) ---
    def update_video_feed(self):
        # ... (Determine active_video_label - unchanged) ...
        active_video_label = None
        if self.current_mode == 'enroll' and hasattr(self, 'enroll_video_label') and self.enroll_video_label.winfo_exists():
            active_video_label = self.enroll_video_label
        elif self.current_mode == 'vote' and hasattr(self, 'vote_video_label') and self.vote_video_label.winfo_exists():
            active_video_label = self.vote_video_label

        # ... (Initialize camera if needed - unchanged) ...
        if self.video_capture is None or not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture or not self.video_capture.isOpened():
                 if active_video_label:
                      placeholder = Image.new('RGB', (320, 240), color = 'grey')
                      imgtk = ImageTk.PhotoImage(image=placeholder)
                      active_video_label.imgtk = imgtk
                      active_video_label.config(image=imgtk, text="Camera Error", compound=tk.CENTER)
                 self._update_status("ERROR: Cannot access camera.", error=True)
                 if self.root.winfo_exists(): self.root.after(1000, self.update_video_feed)
                 return

        # ... (Grab frame - unchanged) ...
        ret, frame = self.video_capture.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            if self.root.winfo_exists(): self.root.after(50, self.update_video_feed)
            return

        display_frame = frame.copy()
        processed_face_this_frame = False

        # --- Face Detection and Mode Logic ---
        if self.facedetect:
            # ... (unchanged face detection and mode-specific logic) ...
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, FACE_CONFIDENCE_THRESHOLD, MIN_NEIGHBORS)

            if len(faces) > 0:
                 processed_face_this_frame = True
                 x, y, w, h = faces[0]

                 # --- Enrollment Mode Logic ---
                 if self.current_mode == 'enroll' and self.is_capturing:
                    # ... (unchanged enrollment capture) ...
                    crop_img = frame[y:y+h, x:x+w]
                    resized_img = cv2.resize(crop_img, FACE_SIZE)
                    if len(self.enroll_faces_data) < FRAMES_TO_CAPTURE:
                        self.enroll_faces_data.append(resized_img)
                        if hasattr(self, 'enroll_progress_label') and self.enroll_progress_label.winfo_exists():
                             progress_text = f"Captured: {len(self.enroll_faces_data)}/{FRAMES_TO_CAPTURE}"
                             self.enroll_progress_label.config(text=progress_text)
                        if len(self.enroll_faces_data) == FRAMES_TO_CAPTURE:
                              self.toggle_capture()
                              speak_async("Face capture complete. Please save.")
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


                 # --- Voting Mode Logic ---
                 elif self.current_mode == 'vote' and self.is_voting_active and self.knn_model:
                    # ... (unchanged voting recognition) ...
                    crop_img = frame[y:y+h, x:x+w]
                    resized_img = cv2.resize(crop_img, FACE_SIZE)
                    flattened_img = resized_img.flatten().reshape(1, -1)
                    try:
                        output = self.knn_model.predict(flattened_img)
                        aadhar = output[0]
                        # ... (streak check, status update, button enable/disable, already voted check - unchanged) ...
                        if aadhar == self.last_prediction_vote: self.prediction_streak_vote += 1
                        else:
                           self._cancel_status_reset_timer()
                           self.last_prediction_vote = aadhar
                           self.prediction_streak_vote = 1
                           self.recognized_aadhar_vote = None
                           self._disable_vote_buttons()

                        if self.prediction_streak_vote >= MIN_PREDICTION_FRAMES_VOTE:
                           self.recognized_aadhar_vote = aadhar
                           display_text = f"ID: {aadhar[-4:]}****"
                           box_color = (0, 255, 0)
                           if aadhar in self.voted_ids:
                               status_msg = f"Voter {aadhar[-4:]}**** ALREADY VOTED."
                               self._update_status(status_msg, error=True)
                               box_color = (0, 0, 255)
                               self._disable_vote_buttons()
                               if self.status_reset_timer_id is None:
                                   speak_async(f"Voter {aadhar} has already voted.")
                                   if self.root.winfo_exists():
                                       self.status_reset_timer_id = self.root.after(RESET_DELAY_MS_VOTE, self._reset_status_after_delay)
                           else:
                               self._cancel_status_reset_timer()
                               status_msg = f"Voter {aadhar[-4:]}**** recognized. Please Vote."
                               self._update_status(status_msg)
                               self._enable_vote_buttons()
                        else:
                            self._cancel_status_reset_timer()
                            display_text = "Identifying..."
                            box_color = (0, 255, 255)
                            self._disable_vote_buttons()
                            if self.status_reset_timer_id is None:
                                self._update_status("Identifying face...")

                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, 2)
                        text_bg_y = max(0, y - 30)
                        cv2.rectangle(display_frame, (x, text_bg_y), (x + w, y), box_color, -1)
                        cv2.putText(display_frame, display_text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    except Exception as e:
                        print(f"Vote Prediction Error: {e}")
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # Blue for error

                 else:
                      # Face detected, but not in active enroll/vote mode
                      cv2.rectangle(display_frame, (x, y), (x+w, y+h), (200, 200, 200), 1)


        # --- Handle No Face Detected or Not in Active Mode ---
        if self.current_mode == 'vote' and not processed_face_this_frame:
            # ... (unchanged reset logic) ...
            self._cancel_status_reset_timer()
            self.last_prediction_vote = None
            self.prediction_streak_vote = 0
            self.recognized_aadhar_vote = None
            self._disable_vote_buttons()
            if self.status_reset_timer_id is None:
                 self._update_status("Place face in front of camera.")


        # --- Update Tkinter Image Label (with dynamic resizing) ---
        if active_video_label and active_video_label.winfo_exists():
            try:
                # 1. Convert original frame to PIL
                cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
                img_pil = Image.fromarray(cv2image)
                img_w, img_h = img_pil.size

                # 2. Get label size
                label_w = active_video_label.winfo_width()
                label_h = active_video_label.winfo_height()

                # 3. Calculate scale (handle invalid label size)
                if label_w > 1 and label_h > 1 and img_w > 0 and img_h > 0:
                    scale = min(label_w / img_w, label_h / img_h)
                    target_w = int(img_w * scale)
                    target_h = int(img_h * scale)
                    # 5. Resize PIL image if needed
                    if target_w > 0 and target_h > 0:
                        img_pil_resized = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    else:
                        img_pil_resized = img_pil # Fallback
                else:
                    img_pil_resized = img_pil # Use original if label size invalid

                # 6. Create PhotoImage and update label
                imgtk = ImageTk.PhotoImage(image=img_pil_resized)
                active_video_label.imgtk = imgtk
                active_video_label.config(image=imgtk)

            except Exception as e_img:
                print(f"Error updating GUI image: {e_img}")
        # --- End Update Tkinter Image Label ---


        # Schedule next update only if the window hasn't been destroyed
        if self.root.winfo_exists():
            is_video_open = self.video_capture and self.video_capture.isOpened()
            if is_video_open:
                self.root.after(30, self.update_video_feed)
            else:
                print("Video source closed unexpectedly.")
                self._update_status("ERROR: Camera disconnected.", error=True)


    # --- Status Update Helper ---
    def _update_status(self, text, error=False):
       if hasattr(self, 'status_label') and self.status_label.winfo_exists():
            self.root.after(0, lambda: self.status_label.config(text=text,
                                                            foreground="red" if error else "black",
                                                            background="lightcoral" if error else "SystemButtonFace"))
       print(f"Status: {text}")


    # --- Cleanup ---
    def on_close(self):
        # ... (unchanged cleanup logic) ...
        print("Closing application...")
        self._cancel_status_reset_timer()
        self.is_capturing = False
        self.is_voting_active = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            print("Camera released.")
        cv2.destroyAllWindows()
        if engine:
            try: engine.stop()
            except: pass
        if self.root:
            try: self.root.destroy()
            except tk.TclError: pass # Ignore error if already destroyed
            print("Tkinter window destroyed.")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVoteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Closing application.")
        app.on_close()