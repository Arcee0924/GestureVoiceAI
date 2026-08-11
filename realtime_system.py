
import cv2
import numpy as np
import tensorflow as tf
import pickle
import time

from gesture_detector import get_landmarks
from speech_engine import speak

# load model
model = tf.keras.models.load_model("models/gesture_model.h5")
encoder = pickle.load(open("models/label_encoder.pkl","rb"))

cap = cv2.VideoCapture(0)

sentence = []
last_word = None

gesture_hold_start = 0
last_detection_time = time.time()

HOLD_TIME = 1.5      # seconds to confirm gesture
SILENCE_TIME = 3     # seconds to speak sentence

while True:

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Camera frame not available.")
        break

    frame, data = get_landmarks(frame)

    current_time = time.time()

    if data:

        prediction = model.predict(np.array([data]), verbose=0)

        confidence = np.max(prediction)
        classID = np.argmax(prediction)

        gesture = encoder.inverse_transform([classID])[0]

        cv2.putText(frame, gesture, (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        if confidence > 0.8:

            if last_word != gesture:
                gesture_hold_start = current_time
                last_word = gesture

            elif current_time - gesture_hold_start > HOLD_TIME:

                sentence.append(gesture)
                print("Added:", gesture)

                last_detection_time = current_time
                last_word = None

    # show sentence on screen
    cv2.putText(frame,
                "Sentence: " + " ".join(sentence),
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(255,0,0),2)

    # speak sentence after inactivity
    if sentence and (current_time - last_detection_time > SILENCE_TIME):

        final_sentence = " ".join(sentence)

        print("Speaking:", final_sentence)

        speak(final_sentence)

        sentence.clear()

        last_word = None
        last_detection_time = current_time

    cv2.imshow("Gesture Voice AI", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()