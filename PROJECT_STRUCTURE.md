# GestureVoiceAI — Project Structure

```text
GestureVoiceAI/
│
├── models/
│   ├── gesture_model.h5       # Pre-trained Keras Neural Network classifier
│   └── label_encoder.pkl      # Pickled Scikit-Learn LabelEncoder for gesture classes
│
├── gesture_images/            # Visual guide assets for gesture manual interface
│   ├── Left_Hand/             # Reference images for left-hand alphabet gestures
│   └── Right_Hand/            # Reference images for right-hand digit & phrase gestures
│
├── data/
│   └── README.md              # Dataset structure specification & collection documentation
│
├── gui.py                     # Primary graphical user interface application (Tkinter)
├── realtime_system.py         # Secondary CLI real-time recognition viewer (OpenCV)
├── gesture_detector.py        # MediaPipe 3D hand landmark detection & drawing module
├── speech_engine.py           # PyTTSX3 offline text-to-speech synthesis wrapper
├── collect_dataset.py         # Landmark data collection script for custom gestures
├── train_model.py             # Keras model training & evaluation script
│
├── requirements.txt           # Python package dependency specification
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git version control exclusions
└── LICENSE                    # Open-source MIT License
```

---

## Directory Description

| Directory / File | Purpose |
| :--- | :--- |
| **`models/`** | Contains the trained Keras HDF5 model (`gesture_model.h5`) and pickled label encoder (`label_encoder.pkl`). |
| **`gesture_images/`** | Stores reference images displayed inside the GUI Gesture Guide interface. |
| **`data/`** | Houses dataset documentation explaining landmark coordinate formatting and dataset collection. |
| **`gui.py`** | Main entry point for the desktop GUI, featuring live camera feed, confidence meters, and TTS output. |
| **`realtime_system.py`** | Standalone OpenCV command-line application for direct real-time gesture recognition. |
| **`gesture_detector.py`** | Extracts 21 3D hand landmark coordinates `(x, y, z)` per frame using MediaPipe Hands. |
| **`speech_engine.py`** | Converts accumulated gesture phrases into voice output using PyTTSX3. |
| **`collect_dataset.py`** | Utility to capture live webcam hand landmarks and log labeled coordinate data to CSV. |
| **`train_model.py`** | Trains the 3-layer Dense Neural Network on landmark feature datasets and exports model artifacts. |
| **`requirements.txt`** | Lists exact library versions required to execute, train, and test the project. |

---

## Entry Point

- **Primary Entry Point:** `gui.py`
  - Started via `python gui.py` (or `.\venv\Scripts\python.exe gui.py`).
  - Launches a Tkinter window featuring navigation screens (Start, Recognition, Gesture Manual) and controls the camera/inference loop.
- **Secondary Entry Point:** `realtime_system.py`
  - Started via `python realtime_system.py`.
  - Runs a lightweight OpenCV video window displaying bounding landmarks and recognized gesture text overlays.

---

## Main Components

1. **Landmark Extractor (`gesture_detector.py`):** Wraps MediaPipe Hands solution to process BGR frames and extract 63 normalized spatial coordinate values.
2. **Gesture Classifier (`models/` & `train_model.py`):** Fully-connected Multi-Layer Perceptron (128 -> 64 -> 47 nodes) trained using Keras with Adam optimizer and Softmax activation.
3. **Text-to-Speech Engine (`speech_engine.py`):** Offline TTS synthesizer utilizing PyTTSX3 to speak compiled sentences upon gesture hold and silence detection.
4. **Graphical Interface (`gui.py`):** Tkinter application providing real-time camera display, top 3 class probability lists, visual confidence progress bar, and user guide.
5. **Data Collection Pipeline (`collect_dataset.py`):** Automated utility script to build custom landmark datasets from live camera frames.
