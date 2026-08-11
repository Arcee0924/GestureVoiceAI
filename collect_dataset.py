import cv2
import mediapipe as mp
import csv
i=0

gesture = input("Enter gesture name: ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

file = open("gesture_dataset.csv", "a", newline="")
writer = csv.writer(file)

print("Show gesture and press S to capture")

while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            data = []

            for lm in hand_landmarks.landmark:

                data.append(lm.x)
                data.append(lm.y)
                data.append(lm.z)

            data.append(gesture)

            writer.writerow(data)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        i=i+1
        print("Sample saved,",i)

    if key == 27:
        break

cap.release()
file.close()
cv2.destroyAllWindows()