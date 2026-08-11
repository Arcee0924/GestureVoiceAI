# GestureVoiceAI — System Architecture

## 1. System Overview

**GestureVoiceAI** is an end-to-end computer vision and speech AI system designed to bridge sign language communication gaps. It captures live video frames via webcam, extracts 21 3D hand landmark spatial coordinates `(x, y, z)` using MediaPipe Hands, classifies hand gestures across 47 target categories using a custom Keras Deep Neural Network, and synthesizes completed phrase sequences into spoken audio using PyTTSX3.

---

## 2. High-Level Architecture

```mermaid
flowchart TD
    User([User / Hand Gesture]) -->|Webcam Video Feed| OpenCV[OpenCV Video Capture]
    OpenCV -->|RGB Frame| Detector[MediaPipe Landmark Extractor]
    Detector -->|63 3D Spatial Coordinates| KerasModel[Keras Dense Neural Network]
    KerasModel -->|Softmax Probabilities| Decoder[Scikit-Learn Label Encoder]
    Decoder -->|Predicted Gesture Class| UI[Tkinter GUI / OpenCV Overlay]
    UI -->|Inactivity Silence Trigger| TTSEngine[PyTTSX3 Speech Engine]
    TTSEngine -->|Audio Waveform| Speaker([Audio Speaker Output])
```

---

## 3. End-to-End Workflow

```mermaid
flowchart LR
    A[Webcam Frame Capture] --> B[BGR to RGB Conversion]
    B --> C[MediaPipe Hand Processing]
    C -->|Hand Detected?| D{Hand Found?}
    D -- Yes --> E[Extract 21 Landmark Coordinates]
    D -- No --> F[Update Silence Timer & Reset State]
    E --> G[Model Predict: 63 Floats -> Softmax]
    G --> H{Confidence > 0.8?}
    H -- Yes --> I[Gesture Hold Time Check]
    I -->|Hold Time > 1.5s| J[Append Gesture to Sentence Buffer]
    H -- No --> K[Display Current Frame Only]
    F --> L{No Hand > 3.0s & Sentence Not Empty?}
    L -- Yes --> M[PyTTSX3 Synthesize & Speak Sentence]
    M --> N[Clear Sentence Buffer]
    L -- No --> K
```

---

## 4. ML / AI Pipeline

```mermaid
flowchart TD
    SubGraph1[Input Processing] --> SubGraph2[Landmark Extraction] --> SubGraph3[Neural Classifier] --> SubGraph4[Post-Processing]

    subgraph SubGraph1 [Input Frame]
        A[Live Webcam Image 480x640x3 BGR]
        B[OpenCV cvtColor -> RGB]
        A --> B
    end

    subgraph SubGraph2 [MediaPipe Hands]
        C[MediaPipe Hands Solution]
        D[21 Keypoints x, y, z Coordinates]
        B --> C --> D
    end

    subgraph SubGraph3 [Keras Sequential NN]
        E[Input Layer: 63 Feature Floats]
        F[Dense Layer 1: 128 Relu Nodes]
        G[Dense Layer 2: 64 Relu Nodes]
        H[Output Layer: 47 Softmax Nodes]
        D --> E --> F --> G --> H
    end

    subgraph SubGraph4 [Classification & Decoding]
        I[Argmax Class ID Selection]
        J[LabelEncoder inverse_transform]
        K[Gesture String Label e.g., 'hello', 'a', '5']
        H --> I --> J --> K
    end
```

### Model Architecture Specification
- **Input Shape:** `(63,)` vector containing 21 MediaPipe hand keypoints `(x, y, z)`.
- **Layer 1:** Fully Connected `Dense(128, activation='relu')`.
- **Layer 2:** Fully Connected `Dense(64, activation='relu')`.
- **Output Layer:** Fully Connected `Dense(47, activation='softmax')`.
- **Loss Function:** `sparse_categorical_crossentropy`.
- **Optimizer:** `adam`.

---

## 5. Application / UI Architecture

```mermaid
graph TD
    UserApp([User Interface]) --> RootWindow[Tkinter Root Window]
    RootWindow --> StartScreen[Start Screen]
    RootWindow --> RecognitionScreen[Recognition Screen]
    RootWindow --> GuideScreen[Gesture Guide Manual Screen]

    RecognitionScreen --> VideoWidget[PIL ImageTk Video Feed Canvas]
    RecognitionScreen --> ConfidenceMeter[Ttk Progressbar Green/Yellow/Red]
    RecognitionScreen --> PredictionPanel[Top 3 Prediction Probabilities]
    RecognitionScreen --> ControlButtons[Start / Clear / Manual / Back]

    VideoWidget --> FrameLoop[10ms Tkinter after Loop]
    FrameLoop --> DetectorModule[gesture_detector.py]
    DetectorModule --> PredictModule[gesture_model.h5 + label_encoder.pkl]
    PredictModule --> SpeechModule[speech_engine.py]
```

---

## 6. Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant Cam as OpenCV VideoCapture
    participant Det as gesture_detector.py
    participant Model as Keras Model & Encoder
    participant GUI as gui.py UI State
    participant TTS as speech_engine.py

    Cam->>Det: Raw Frame Array (480x640 BGR)
    Det->>Det: Convert RGB & Run MediaPipe Hands
    Det-->>GUI: Annotated Frame + 63-Float Array
    alt Hand Detected
        GUI->>Model: np.array([63 Landmark Floats])
        Model-->>GUI: Softmax Probabilities & Predicted Label
        GUI->>GUI: Update Progress Bar & Top 3 Predictions
        opt Gesture Held > 1.5 Seconds
            GUI->>GUI: Append Label to Sentence String
        end
    else No Hand Detected
        GUI->>GUI: Increment Inactivity Timer
        opt Inactivity Silence > 3 Seconds
            GUI->>TTS: speak(accumulated_sentence)
            TTS-->>GUI: Audio Playback Complete & Reset Sentence
        end
    end
```

---

## 7. Training Pipeline

```mermaid
flowchart TD
    A[gesture_dataset.csv Raw Data] -->|63 Features + 1 String Label| B[Pandas DataFrame Read]
    B --> C[Extract Features X: 63 Landmark Floats]
    B --> D[Extract Labels y: Target Strings]
    D --> E[Scikit-Learn LabelEncoder Fit & Transform]
    C --> F[Train / Test Split 80% Train, 20% Test]
    E --> F
    F --> G[Build Keras Sequential Model 128-64-47]
    G --> H[Compile: Adam + Sparse Categorical Crossentropy]
    H --> I[Model Fit: 30 Epochs, Batch Size 32]
    I --> J[Evaluate Test Accuracy]
    J --> K[Export models/gesture_model.h5]
    J --> L[Export models/label_encoder.pkl via Pickle]
```

---

## 8. Technology Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10 |
| **Deep Learning Framework** | TensorFlow / Keras (v2.10.0) |
| **Hand Tracking / Landmark Extraction** | MediaPipe Hands (v0.10.9) |
| **Computer Vision & Image Processing** | OpenCV (`opencv-python` v4.6.0) |
| **User Interface Framework** | Tkinter & PIL (`Pillow` v9.5.0) |
| **Speech Synthesis (TTS)** | PyTTSX3 (v2.90) |
| **Data Processing & ML Utilities** | NumPy (v1.23.5), Pandas (v1.5.0), Scikit-Learn (v1.1.3) |
| **Model Serialization Format** | HDF5 (`.h5`) & Pickle (`.pkl`) |

---

## 9. Component Relationships

```mermaid
classDiagram
    class GuiApp {
        +start_system()
        +stop_system()
        +update_frame()
        +clear_sentence()
        +on_closing()
    }
    class RealtimeSystem {
        +main_loop()
    }
    class GestureDetector {
        +get_landmarks(frame)
    }
    class SpeechEngine {
        +speak(text)
    }
    class ModelArtifacts {
        +gesture_model.h5
        +label_encoder.pkl
    }
    class DatasetCollector {
        +capture_samples()
    }
    class ModelTrainer {
        +train_and_evaluate()
    }

    GuiApp --> GestureDetector : Imports & Calls
    GuiApp --> SpeechEngine : Imports & Calls
    GuiApp --> ModelArtifacts : Loads at Startup

    RealtimeSystem --> GestureDetector : Imports & Calls
    RealtimeSystem --> SpeechEngine : Imports & Calls
    RealtimeSystem --> ModelArtifacts : Loads at Startup

    DatasetCollector --> GestureDetector : Uses for Data Collection
    ModelTrainer --> ModelArtifacts : Trains & Exports
```

---

## 10. Complete Architecture Summary

The GestureVoiceAI system provides a complete real-time sign-language recognition and speech generation solution. Live webcam frames are ingested by OpenCV and converted to RGB space before passing to MediaPipe Hands, which extracts 21 keypoint 3D spatial coordinates `(x, y, z)`. The resulting 63-element feature vector is fed into a trained 3-layer Keras Dense Neural Network to compute softmax class probabilities across 47 hand gesture categories. A Scikit-Learn LabelEncoder resolves the predicted index to its target text string. The Tkinter GUI tracks gesture stability over a 1.5-second hold threshold to construct sentences, and after 3 seconds of hand inactivity, the PyTTSX3 speech engine speaks the accumulated sentence aloud before resetting state.
