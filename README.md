# GestureVoiceAI 🖐️🔊

A real-time Computer Vision and Speech AI system that translates hand gestures into text and spoken audio using MediaPipe hand tracking, a TensorFlow/Keras neural network classifier, and PyTTSX3 text-to-speech synthesis.

---

## 🌟 Key Features

- **Real-Time Hand Tracking:** Extracts 21 3D hand landmarks `(x, y, z)` per frame using MediaPipe.
- **Gesture Classification:** Dense Neural Network model classifying 47 distinct hand gestures (alphabets `a-z`, digits `0-9`, and common phrases like `hello`, `thankyou`, `yes`, `no`, `please`, `eat`, `sleep`).
- **Text-to-Speech Engine:** Automatically speaks composed sentences upon gesture completion and silence detection using PyTTSX3.
- **Modern Graphical User Interface:** Built with Tkinter featuring real-time camera display, live confidence progress bars, top gesture predictions, and a visual gesture guide.
- **Dataset Collection & Training:** Built-in tools for collecting customized hand landmark data and retraining the gesture classifier.

---

## 📁 Project Structure

```
GestureVoiceAI/
│
├── models/
│   ├── gesture_model.h5       # Trained Keras neural network model
│   └── label_encoder.pkl      # Scikit-learn LabelEncoder for gesture classes
│
├── gesture_images/            # Reference guide images for gestures
│   ├── Left_Hand/
│   └── Right_Hand/
│
├── data/
│   └── README.md              # Dataset format & collection instructions
│
├── gui.py                     # Primary Tkinter GUI entry point
├── realtime_system.py         # OpenCV real-time CLI entry point
├── gesture_detector.py        # MediaPipe landmark extraction module
├── speech_engine.py           # PyTTSX3 text-to-speech synthesis module
├── collect_dataset.py         # Data collection script for new gestures
├── train_model.py             # Neural network model training script
├── requirements.txt           # Python dependency requirements
├── .gitignore                 # Git ignore rules
└── README.md                  # Project documentation
```

---

## ⚙️ Requirements & Setup

### Prerequisites
- Python 3.10+
- Webcam

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Arcee0924/GestureVoiceAI.git
   cd GestureVoiceAI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Graphical User Interface (GUI App)
Run the main application:
```bash
python gui.py
```
- Click **Start Recognition** to begin webcam gesture tracking.
- Hold a gesture steady to append it to the active sentence.
- Pause to hear the synthesized speech output.
- Access the **Gesture Guide** to view supported hand gestures.

### Real-time CLI Mode
Run the lightweight OpenCV window:
```bash
python realtime_system.py
```
- Press `ESC` to exit the application.

---

## 📊 Dataset Collection & Retraining

1. **Collect Data:**
   ```bash
   python collect_dataset.py
   ```
   Enter the gesture label name, show your hand in front of the camera, and press `S` to save landmark coordinates.

2. **Train Model:**
   ```bash
   python train_model.py
   ```
   Trains the neural network on `gesture_dataset.csv` and saves the updated model to `models/gesture_model.h5` and `models/label_encoder.pkl`.

---

## 📜 License
This project is open-source under the MIT License.
