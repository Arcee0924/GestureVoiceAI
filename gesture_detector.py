import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils


def get_landmarks(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    data = None

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            data = []

            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)
                data.append(lm.z)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return frame, data