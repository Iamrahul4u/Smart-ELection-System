# server.py
import base64
import io
import eventlet # Recommended for flask-socketio concurrency
eventlet.monkey_patch() # Important! Patches standard libraries for concurrency

import threading
import time
from datetime import datetime
import os
import pickle
import csv
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from sklearn.neighbors import KNeighborsClassifier

# --- Configuration ---
DATA_DIR = 'data/'
FACES_PKL = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL = os.path.join(DATA_DIR, 'names.pkl')
MODEL_PKL = os.path.join(DATA_DIR, 'model.pkl')
VOTES_CSV = os.path.join(DATA_DIR, 'Votes.csv')
CASCADE_PATH = os.path.join('cascades', 'haarcascade_frontalface_default.xml') # Ensure this path is correct
FACE_SIZE = (50, 50)
FRAMES_TO_CAPTURE = 50
FACE_CONFIDENCE_THRESHOLD = 1.3 # ScaleFactor for detectMultiScale
MIN_NEIGHBORS = 5             # MinNeighbors for detectMultiScale
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
         os.makedirs(dir_path, exist_ok=True)
    elif not dir_path and not os.path.exists(path) and '.' not in os.path.basename(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)

def validate_aadhar(aadhar_str):
     return aadhar_str is not None and aadhar_str.isdigit() and len(aadhar_str) == 12

# --- Global Variables & Locks ---
client_states = {}
model_lock = threading.Lock()
data_lock = threading.Lock()
train_lock = threading.Lock()
training_in_progress = False

# --- Face Recognition & Data Handling Logic ---
facedetect = None
knn_model = None
voted_ids = set()

def load_resources():
    """Loads cascade classifier, KNN model, and voted IDs."""
    global facedetect, knn_model, voted_ids
    print("Loading resources...")
    ensure_dir_exists(DATA_DIR)
    ensure_dir_exists(os.path.dirname(CASCADE_PATH))

    if not os.path.exists(CASCADE_PATH):
        print(f"ERROR: Cascade file not found: {CASCADE_PATH}")
        facedetect = None
    else:
        facedetect = cv2.CascadeClassifier(CASCADE_PATH)
        if facedetect.empty():
            print(f"ERROR: Failed to load cascade classifier from: {CASCADE_PATH}")
            facedetect = None
        else:
            print("Face detector loaded successfully.")

    _load_model_internal()
    _load_voted_ids_internal()

def _load_model_internal():
    """Loads the KNN model from pickle file safely."""
    global knn_model
    with model_lock:
        if not os.path.exists(MODEL_PKL):
            print(f"Warning: Model file {MODEL_PKL} not found. Model needs training.")
            knn_model = None; return False
        try:
            if os.path.getsize(MODEL_PKL) == 0:
                print(f"Warning: Model file {MODEL_PKL} exists but is empty."); knn_model = None; return False
            with open(MODEL_PKL, 'rb') as f: knn_model = pickle.load(f)
            print(f"KNN Model loaded successfully from {MODEL_PKL}."); return True
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error: Model file {MODEL_PKL} is empty or corrupted: {e}"); knn_model = None; return False
        except Exception as e:
            print(f"Error loading KNN model from {MODEL_PKL}: {e}"); knn_model = None; return False

def _load_voted_ids_internal():
    """Loads the set of already voted Aadhar numbers from CSV safely."""
    global voted_ids
    with data_lock:
        voted_ids = set()
        if os.path.exists(VOTES_CSV):
            try:
                with open(VOTES_CSV, "r", newline='') as csvfile:
                    reader = csv.reader(csvfile); header = next(reader, None) # Skip header
                    if header and header[0].strip().upper() != COL_NAMES_VOTES[0]: print(f"Warning: Votes.csv header mismatch.")
                    for i, row in enumerate(reader):
                        if row and len(row) > 0: voted_ids.add(row[0].strip())
                        else: print(f"Warning: Empty row found at index {i+1} in {VOTES_CSV}")
                print(f"Loaded {len(voted_ids)} previously voted IDs from {VOTES_CSV}.")
            except Exception as e:
                print(f"Error reading {VOTES_CSV}: {e}. Starting with empty voted list."); voted_ids = set()
        else: print(f"{VOTES_CSV} not found. Starting with empty voted list.")

def _check_aadhar_exists_internal(aadhar_to_check):
    """Checks if an Aadhar number exists in the names.pkl file safely."""
    with data_lock:
        if not os.path.exists(NAMES_PKL): return False
        try:
            if os.path.getsize(NAMES_PKL) > 0:
                with open(NAMES_PKL, 'rb') as f: loaded_names = pickle.load(f)
                if isinstance(loaded_names, (list, set)): return aadhar_to_check in loaded_names
                else: print(f"Warning: Expected list/set in {NAMES_PKL}, found {type(loaded_names)}."); return False
            else: return False
        except Exception as e: print(f"Warning checking existing names in {NAMES_PKL}: {e}"); return False

def _record_vote_file_internal(aadhar, party):
    """Records a vote in the CSV file safely."""
    global voted_ids
    with data_lock:
        if aadhar in voted_ids: print(f"Vote blocked: {aadhar} already voted."); return False, "Already Voted"
        ts = time.time(); date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y"); timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        vote_record = [aadhar, party, date, timestamp]
        try:
            ensure_dir_exists(VOTES_CSV); file_exists = os.path.isfile(VOTES_CSV)
            write_header = not file_exists or os.path.getsize(VOTES_CSV) < 10
            with open(VOTES_CSV, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header: writer.writerow(COL_NAMES_VOTES)
                writer.writerow(vote_record)
            voted_ids.add(aadhar); print(f"Vote recorded: {vote_record}"); return True, "Vote Recorded Successfully"
        except PermissionError as e: err_msg = f"Permission denied writing to {VOTES_CSV}: {e}"; print(err_msg); return False, err_msg
        except Exception as e: err_msg = f"Failed to write vote to {VOTES_CSV}: {e}"; print(err_msg); return False, err_msg

def _save_enrollment_internal(sid, aadhar, faces_data_list):
    """Saves captured face data to pickle files safely."""
    with data_lock:
        if not aadhar or not faces_data_list: return False, "No enrollment Aadhar or face data provided."
        if len(faces_data_list) < FRAMES_TO_CAPTURE // 2: print(f"Warning: Only {len(faces_data_list)} frames for {aadhar}. Saving anyway.")
        try:
            if not isinstance(faces_data_list, list) or len(faces_data_list) == 0: return False, "Invalid face data format."
            faces_data_np = np.asarray(faces_data_list); n_samples = faces_data_np.shape[0]
            if faces_data_np.ndim != 4 or faces_data_np.shape[1:3] != FACE_SIZE:
                print(f"Warning: Unexpected shape after converting data: {faces_data_np.shape}.")
                try: faces_data_np = faces_data_np.reshape(n_samples, FACE_SIZE[0], FACE_SIZE[1], -1); print(f"Reshaped to: {faces_data_np.shape}"); assert faces_data_np.ndim == 4
                except Exception as reshape_err: return False, f"Could not process face data shape: {reshape_err}"
            faces_data_flattened = faces_data_np.reshape(n_samples, -1); num_features = faces_data_flattened.shape[1]
            names = []; faces = np.array([]).reshape(0, num_features) # Initialize with correct features
            if os.path.exists(NAMES_PKL) and os.path.getsize(NAMES_PKL) > 0:
                try:
                    with open(NAMES_PKL, 'rb') as f: names = pickle.load(f)
                    if not isinstance(names, list): print(f"Warning: Loaded names not list. Resetting."); names = []
                except Exception as e: print(f"Warning: Could not load {NAMES_PKL}: {e}"); names = []
            if os.path.exists(FACES_PKL) and os.path.getsize(FACES_PKL) > 0:
                try:
                    with open(FACES_PKL, 'rb') as f: faces = pickle.load(f)
                    if not isinstance(faces, np.ndarray) or faces.ndim != 2: raise ValueError(f"Existing faces incorrect type/dim.")
                    if faces.size > 0 and faces.shape[1] != num_features: raise ValueError(f"Shape mismatch! Existing:{faces.shape[1]} != New:{num_features}")
                except Exception as e: print(f"Warning: Could not load/validate {FACES_PKL}: {e}"); faces = np.array([]).reshape(0, num_features)
            names.extend([aadhar] * n_samples)
            faces = faces_data_flattened if faces.size == 0 else np.append(faces, faces_data_flattened, axis=0)
            ensure_dir_exists(NAMES_PKL); ensure_dir_exists(FACES_PKL)
            with open(NAMES_PKL, 'wb') as f: pickle.dump(names, f)
            with open(FACES_PKL, 'wb') as f: pickle.dump(faces, f)
            print(f"Enrollment saved for {aadhar}. Total samples: {len(names)}. Features shape: {faces.shape}")
            return True, f"Enrollment saved for {aadhar[-4:]}****. Train Model Recommended."
        except ValueError as ve: print(f"Save Error (ValueError): {ve}"); return False, f"Data format/shape error: {ve}"
        except Exception as e: print(f"Save Error (Other): {e}"); return False, f"Failed to save enrollment data: {e}"

def _train_model_task():
    """Trains the KNN model using data from pickle files (runs in background thread)."""
    global training_in_progress, knn_model
    if not train_lock.acquire(blocking=False): print("Training already in progress."); socketio.emit('training_status', {'status': 'busy', 'message': 'Training already in progress.'}); return
    training_in_progress = True; print("Starting model training task..."); socketio.emit('training_status', {'status': 'started', 'message': 'Model training started...'})
    success = False; message = "Training failed."; loaded_faces = None; loaded_names = None
    try:
        with data_lock:
            if not os.path.exists(FACES_PKL) or not os.path.exists(NAMES_PKL): raise FileNotFoundError("Data files not found.")
            if os.path.getsize(FACES_PKL) == 0 or os.path.getsize(NAMES_PKL) == 0: raise ValueError("Data files are empty.")
            with open(FACES_PKL, 'rb') as f: loaded_faces = pickle.load(f)
            with open(NAMES_PKL, 'rb') as f: loaded_names = pickle.load(f)
        if not isinstance(loaded_faces, np.ndarray) or loaded_faces.ndim != 2: raise ValueError("Loaded face data invalid.")
        if not isinstance(loaded_names, list): raise ValueError("Loaded name data not list.")
        if loaded_faces.shape[0] != len(loaded_names): raise ValueError(f"Data mismatch! Samples:{loaded_faces.shape[0]}, Labels:{len(loaded_names)}")
        if loaded_faces.size == 0: raise ValueError("No face data found.")
        num_samples = loaded_faces.shape[0]; num_unique_ids = len(np.unique(loaded_names))
        print(f"Training KNN model on {num_samples} samples for {num_unique_ids} unique IDs...")
        temp_knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_KNN, weights='distance', metric='euclidean')
        temp_knn.fit(loaded_faces, loaded_names)
        with model_lock:
            print("Saving trained model..."); ensure_dir_exists(MODEL_PKL)
            with open(MODEL_PKL, 'wb') as f: pickle.dump(temp_knn, f)
            knn_model = temp_knn; message = f"Model training complete ({num_samples} samples) and saved!" ; success = True; print(message)
    except (FileNotFoundError, ValueError) as e: message = f"ERROR: {e}"; success = False
    except Exception as e: message = f"Unexpected training error: {e}"; success = False; print(f"Training Error: {e}")
    finally:
        print(f"Training Worker Finished: Success={success}, Message={message}")
        training_in_progress = False; train_lock.release()
        socketio.emit('training_status', {'status': 'completed' if success else 'failed', 'message': message, 'success': success })

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_replace_me!')
CORS(app, resources={r"/*": {"origins": "*"}}) # Allows all origins for dev
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new client connections."""
    sid = request.sid; print(f"Client connected: {sid}")
    client_states[sid] = { 'mode': 'idle', 'enroll_aadhar': None, 'enroll_faces_data': [], 'capture_count': 0, 'recognized_aadhar': None, 'last_prediction': None, 'prediction_streak': 0 }
    emit('connection_ack', {'message': 'Connected to backend server', 'sid': sid})
    with model_lock: emit('model_status', {'loaded': knn_model is not None})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    sid = request.sid; print(f"Client disconnected: {sid}")
    if sid in client_states: del client_states[sid]; print(f"Cleaned up state for client {sid}")

# server.py

@socketio.on('set_mode')
def handle_set_mode(data):
    """Allows client to request a mode change (optional)."""
    sid = request.sid
    mode = data.get('mode')
    # **** ADD LOG ****
    print(f"--- Received set_mode event from {sid}: Mode requested = {mode} ---")
    # **** END LOG ****
    if sid in client_states:
        current_server_mode = client_states[sid]['mode']
        if current_server_mode != mode:
              # **** ADD LOG ****
              print(f"Server updating mode for {sid} from '{current_server_mode}' to '{mode}'")
              # **** END LOG ****
              client_states[sid]['mode'] = mode
              # Reset specific state parts based on the *new* mode
              if mode != 'enroll_capture': client_states[sid]['enroll_aadhar'] = None; client_states[sid]['enroll_faces_data'] = []; client_states[sid]['capture_count'] = 0
              if mode != 'vote_recognize': client_states[sid]['recognized_aadhar'] = None; client_states[sid]['last_prediction'] = None; client_states[sid]['prediction_streak'] = 0
              emit('mode_set', {'mode': mode}, room=sid) # Confirm back to client
        else: print(f"Client {sid} requested mode {mode}, backend already in that mode."); emit('mode_set', {'mode': mode}, room=sid)
    else: print(f"Error: Client state not found for sid {sid} during set_mode."); emit('error', {'message': 'Server state error: Client not found.'}, room=sid)

# **** Also add logs to handle_video_frame as suggested before ****
@socketio.on('video_frame')
def handle_video_frame(data):
    sid = request.sid
    print(f"--- Received video_frame event from {sid} ---") # Check if triggered
    if sid not in client_states:
         print(f"Error: Client state not found for {sid} in handle_video_frame.")
         return
    state = client_states[sid]
    mode = state.get('mode')
    print(f"DEBUG handle_video_frame: Client {sid} current backend mode is '{mode}'") # Check the mode
    if mode not in ['enroll_capture', 'vote_recognize'] or not facedetect:
         print(f"DEBUG handle_video_frame: Ignoring frame. Mode='{mode}', Facedetect loaded={facedetect is not None}")
         return
    # ... rest of function ...
@socketio.on('start_enroll')
def handle_start_enroll(data):
    """Handles the start of the enrollment process for a given Aadhar."""
    sid = request.sid; aadhar = data.get('aadhar')
    print(f"Server received 'start_enroll' request from {sid} for Aadhar: {aadhar}")
    if not validate_aadhar(aadhar): print(f"Invalid Aadhar format from {sid}: {aadhar}"); emit('enroll_status', {'success': False, 'message': 'Invalid Aadhar number format (must be 12 digits).'}, room=sid); return
    if _check_aadhar_exists_internal(aadhar): print(f"Aadhar {aadhar} already exists, rejected for {sid}."); emit('enroll_status', {'success': False, 'message': f'Aadhar ending ...{aadhar[-4:]} is already enrolled.'}, room=sid); return
    if sid in client_states:
        client_states[sid]['mode'] = 'enroll_capture'; client_states[sid]['enroll_aadhar'] = aadhar; client_states[sid]['enroll_faces_data'] = []; client_states[sid]['capture_count'] = 0
        print(f"Server state updated for {sid}: mode=enroll_capture, aadhar={aadhar}")
        emit('enroll_status', {'success': True, 'message': f'Server ready for enrollment capture for ...{aadhar[-4:]}.', 'aadhar': aadhar, 'frames_needed': FRAMES_TO_CAPTURE }, room=sid)
        print(f"Sent enrollment ready confirmation to {sid}")
    else: print(f"Error: Client state not found for sid {sid} during start_enroll."); emit('error', {'message': 'Server state error: Client session not found.'}, room=sid)

@socketio.on('video_frame')
def handle_video_frame(data):
    """Processes incoming video frames for enrollment or voting."""
    sid = request.sid
    if sid not in client_states: return

    state = client_states[sid]
    mode = state.get('mode')
    if mode not in ['enroll_capture', 'vote_recognize'] or not facedetect: return

    try:
        image_data_url = data.get('image_data_url');
        if not image_data_url or ',' not in image_data_url: raise ValueError("Invalid image data URL")
        image_data = image_data_url.split(',')[1]; image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e: print(f"Error decoding frame from {sid}: {e}"); emit('error', {'message': f'Server failed to decode frame: {e}'}, room=sid); return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=FACE_CONFIDENCE_THRESHOLD, minNeighbors=MIN_NEIGHBORS, minSize=(30, 30))

    processed_face_this_frame = False; bounding_box = None
    if len(faces) > 0:
        processed_face_this_frame = True
        x, y, w, h = faces[0]
        bounding_box = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} # Convert to standard int
        crop_img = frame[y:y+h, x:x+w]
        if crop_img.size == 0: processed_face_this_frame = False; print(f"Warning: Empty face crop for {sid}")
        else:
            resized_img = cv2.resize(crop_img, FACE_SIZE)
            # --- Enrollment Capture Mode ---
            if mode == 'enroll_capture' and state.get('enroll_aadhar'):
                if state['capture_count'] < FRAMES_TO_CAPTURE:
                    state['enroll_faces_data'].append(resized_img)
                    state['capture_count'] += 1; current_count = state['capture_count']
                    emit('enroll_progress', {'count': current_count, 'total': FRAMES_TO_CAPTURE, 'box': bounding_box}, room=sid)
                    if current_count == FRAMES_TO_CAPTURE:
                        print(f"Enrollment capture complete for {sid}, Aadhar {state['enroll_aadhar']}. Attempting to emit ready_to_save.")
                        try:
                            emit('enroll_status', {
                                 'success': True, 'message': f'Collected {FRAMES_TO_CAPTURE} frames. Ready to save.', 'ready_to_save': True
                             }, room=sid)
                            print(f"Successfully emitted ready_to_save for {sid}.")
                        except Exception as e_emit:
                            print(f"ERROR emitting enroll_status with ready_to_save for {sid}: {e_emit}")

            # --- Voting Recognition Mode ---
            elif mode == 'vote_recognize':
                 # **** ADDED DETAILED LOGGING FOR VOTING ****
                 print(f"--- Vote Recognize Frame for {sid} ---")
                 with model_lock:
                     if not knn_model:
                         print(f"ERROR vote_recognize: knn_model is None for {sid}")
                         emit('vote_status', {'status': 'error', 'message': 'Recognition model not loaded on server.'}, room=sid)
                         state['mode'] = 'idle'; return

                     # print(f"DEBUG vote_recognize: knn_model IS available for {sid}") # Uncomment if needed
                     flattened_img = resized_img.flatten().reshape(1, -1)
                     try:
                         output = knn_model.predict(flattened_img); predicted_aadhar = output[0]
                         print(f"DEBUG vote_recognize: Predicted Aadhar: {predicted_aadhar}")

                         last_pred = state.get('last_prediction'); streak = state.get('prediction_streak', 0)
                         if predicted_aadhar == last_pred: streak += 1
                         else: streak = 1; state['recognized_aadhar'] = None
                         state['last_prediction'] = predicted_aadhar; state['prediction_streak'] = streak
                         print(f"DEBUG vote_recognize: Current Streak: {streak}/{MIN_PREDICTION_FRAMES_VOTE}")

                         recognition_status = 'identifying'; recognized_id_display = None; can_vote = False
                         message = f"Identifying ({streak}/{MIN_PREDICTION_FRAMES_VOTE})..."; already_voted = False

                         if streak >= MIN_PREDICTION_FRAMES_VOTE:
                             state['recognized_aadhar'] = predicted_aadhar; recognized_id_display = f"{predicted_aadhar[-4:]}****"
                             print(f"DEBUG vote_recognize: Streak met. Confirmed Aadhar: {predicted_aadhar}")
                             with data_lock:
                                 if predicted_aadhar in voted_ids:
                                     recognition_status = 'already_voted'; message = f"Voter {recognized_id_display} ALREADY VOTED."; can_vote = False; already_voted = True
                                     print(f"DEBUG vote_recognize: Voter {predicted_aadhar} already voted.")
                                 else:
                                     recognition_status = 'recognized'; message = f"Voter {recognized_id_display} recognized. Ready to Vote."; can_vote = True; already_voted = False
                                     print(f"DEBUG vote_recognize: Voter {predicted_aadhar} recognized, CAN VOTE.")
                         else: state['recognized_aadhar'] = None; can_vote = False

                         emit_data = { 'status': recognition_status, 'message': message, 'aadhar_display': recognized_id_display, 'can_vote': can_vote, 'already_voted': already_voted, 'box': bounding_box }
                         print(f"DEBUG vote_recognize: Emitting vote_status: {emit_data}")
                         emit('vote_status', emit_data, room=sid)

                     except Exception as e:
                         print(f"ERROR vote_recognize: Prediction Error for {sid}: {e}")
                         emit('vote_status', { 'status': 'error', 'message': f'Recognition error: {e}', 'box': bounding_box }, room=sid)
                 # **** END DETAILED LOGGING FOR VOTING ****

    # --- Handle No Face Detected ---
    if not processed_face_this_frame:
        if mode == 'vote_recognize':
             if state.get('last_prediction') is not None or state.get('recognized_aadhar') is not None:
                 state['last_prediction'] = None; state['prediction_streak'] = 0; state['recognized_aadhar'] = None
                 emit('vote_status', { 'status': 'no_face', 'message': 'Place face in camera view.', 'can_vote': False }, room=sid)
        elif mode == 'enroll_capture':
              if sid in client_states: emit('enroll_progress', { 'count': state.get('capture_count', 0), 'total': FRAMES_TO_CAPTURE, 'box': None }, room=sid)

@socketio.on('save_enrollment')
def handle_save_enrollment():
    """Handles request from client to save captured enrollment data."""
    sid = request.sid; print(f"Save enrollment requested by {sid}")
    if sid not in client_states: emit('error', {'message': 'Client state not found.'}, room=sid); return
    state = client_states[sid]; aadhar = state.get('enroll_aadhar'); faces_data = state.get('enroll_faces_data')
    if not aadhar or not faces_data: emit('enroll_save_status', {'success': False, 'message': 'Missing Aadhar or face data on server.'}, room=sid); return
    if len(faces_data) < FRAMES_TO_CAPTURE: print(f"Warning: Save requested by {sid} but only {len(faces_data)}/{FRAMES_TO_CAPTURE} frames found.")
    success, message = _save_enrollment_internal(sid, aadhar, faces_data)
    emit('enroll_save_status', {'success': success, 'message': message}, room=sid)
    if success:
        print(f"Enrollment save successful for {sid}, resetting state.")
        state['mode'] = 'idle'; state['enroll_aadhar'] = None; state['enroll_faces_data'] = []; state['capture_count'] = 0
        with model_lock: emit('model_status', {'loaded': knn_model is not None})
    else: print(f"Enrollment save failed for {sid}.")

@socketio.on('cast_vote')
def handle_cast_vote(data):
    """Handles request from client to cast a vote."""
    sid = request.sid; party = data.get('party'); print(f"Vote cast request from {sid} for party: {party}")
    if sid not in client_states: emit('error', {'message': 'Client state not found.'}, room=sid); return
    state = client_states[sid]; aadhar_to_vote = state.get('recognized_aadhar')
    if not aadhar_to_vote: print(f"Vote rejected for {sid}: No voter recognized."); emit('vote_result', {'success': False, 'message': 'No voter recognized or recognition not stable.'}, room=sid); return
    if not party or party not in PARTIES: print(f"Vote rejected for {sid}: Invalid party '{party}'."); emit('vote_result', {'success': False, 'message': f'Invalid party selection: {party}.'}, room=sid); return
    success, message = _record_vote_file_internal(aadhar_to_vote, party)
    emit('vote_result', { 'success': success, 'message': message, 'party': party, 'aadhar_display': f"{aadhar_to_vote[-4:]}****" if success else None }, room=sid)
    if success: print(f"Vote recorded for {sid}, Aadhar {aadhar_to_vote}. Resetting state."); state['mode'] = 'idle'; state['recognized_aadhar'] = None; state['last_prediction'] = None; state['prediction_streak'] = 0
    else: print(f"Vote recording failed for {sid}, Aadhar {aadhar_to_vote}. Resetting state."); state['recognized_aadhar'] = None; state['last_prediction'] = None; state['prediction_streak'] = 0

@socketio.on('train_model')
def handle_train_model():
    """Handles request from client to start model training."""
    sid = request.sid; print(f"Train model request received from client {sid}.")
    if training_in_progress: print("Training busy, rejecting request."); emit('training_status', {'status': 'busy', 'message': 'Training already in progress.'}, room=sid); return
    print("Starting training task in background thread."); train_thread = threading.Thread(target=_train_model_task, daemon=True); train_thread.start()

# --- Load initial resources at startup ---
load_resources()

# --- Main Execution ---
if __name__ == '__main__':
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask-SocketIO server on {host}:{port}...")
    socketio.run(app, host=host, port=port, use_reloader=False)