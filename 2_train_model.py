# 2_train_model.py
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuration ---
DATA_DIR = 'data/'
FACES_PKL = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PKL = os.path.join(DATA_DIR, 'names.pkl')
MODEL_PKL = os.path.join(DATA_DIR, 'model.pkl')
N_NEIGHBORS = 5

# --- Main Logic ---
print("Loading data...")
if not os.path.exists(FACES_PKL) or not os.path.exists(NAMES_PKL):
    print(f"ERROR: Data files not found in '{DATA_DIR}'.")
    print("Please run '1_add_faces.py' first.")
    exit()

try:
    with open(FACES_PKL, 'rb') as f:
        faces = pickle.load(f)

    with open(NAMES_PKL, 'rb') as f:
        names = pickle.load(f)

    if len(faces) != len(names):
        print("ERROR: Number of faces and names do not match!")
        print(f"Faces: {len(faces)}, Names: {len(names)}")
        exit()

    if len(faces) == 0:
        print("ERROR: No face data found.")
        exit()

    print(f"Loaded {len(faces)} face samples for {len(np.unique(names))} unique individuals.")
    print(f"Data shape: {faces.shape}")

    # Ensure faces data is 2D (n_samples, n_features)
    if faces.ndim != 2:
        print(f"ERROR: Faces data has incorrect dimensions: {faces.ndim}. Expected 2.")
        # Attempt reshape if possible (assuming original was (n_samples, height, width, channels))
        # This depends heavily on how data was saved in add_faces.py
        # If add_faces saved flattened data, this isn't needed.
        # faces = faces.reshape(len(faces), -1)
        # print(f"Attempted reshape to: {faces.shape}")
        exit() # Safer to exit if unsure about shape

    print("\nTraining KNN model...")
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance', metric='euclidean')
    knn.fit(faces, names)
    print("Training complete.")

    # --- Optional: Evaluate Model ---
    if len(np.unique(names)) > 1 and len(faces) > 10: # Need enough data for split
        print("\nEvaluating model accuracy (on training data)...")
        # Split data for a simple evaluation (not a robust validation)
        try:
             X_train, X_test, y_train, y_test = train_test_split(faces, names, test_size=0.25, stratify=names, random_state=42)
             knn_eval = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights='distance', metric='euclidean')
             knn_eval.fit(X_train, y_train)
             y_pred = knn_eval.predict(X_test)
             accuracy = accuracy_score(y_test, y_pred)
             print(f"Model Accuracy (estimated on split): {accuracy:.2f}")
        except Exception as e:
             print(f"Could not perform evaluation split: {e}")


    print(f"\nSaving trained model to {MODEL_PKL}...")
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump(knn, f)
    print("Model saved successfully.")

except FileNotFoundError:
     print(f"ERROR: Could not find data files in {DATA_DIR}")
except Exception as e:
    print(f"An error occurred: {e}")