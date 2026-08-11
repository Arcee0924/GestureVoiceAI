import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import pickle
import time

from gesture_detector import get_landmarks
from speech_engine import speak


# ===============================
# LOAD MODEL (UNCHANGED)
# ===============================

model = tf.keras.models.load_model("models/gesture_model.h5")
encoder = pickle.load(open("models/label_encoder.pkl","rb"))

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

no_hand_start = None
# sentence = []
sentence = ""
last_word = None
gesture_hold_start = 0
last_detection_time = time.time()

HOLD_TIME = 1.5
SILENCE_TIME = 3

running = False


# ===============================
# SCREEN FUNCTIONS
# ===============================

def show_start():
    recognition_frame.pack_forget()
    manual_frame.pack_forget()
    add_frame.pack_forget()
    start_frame.pack(fill="both", expand=True)


def show_recognition():
    start_frame.pack_forget()
    manual_frame.pack_forget()
    add_frame.pack_forget()
    recognition_frame.pack(fill="both", expand=True)


def show_manual():
    start_frame.pack_forget()
    recognition_frame.pack_forget()
    add_frame.pack_forget()
    manual_frame.pack(fill="both", expand=True)


def show_add():
    start_frame.pack_forget()
    recognition_frame.pack_forget()
    manual_frame.pack_forget()
    add_frame.pack(fill="both", expand=True)


# ===============================
# AI SYSTEM (UNCHANGED)
# ===============================
# the change is made
# def start_system():
#     global running
#     running = True
#     update_frame()
def start_system():
    global running
    if not running:
        running = True
        update_frame()


def stop_system():
    global running
    running = False


def clear_sentence():
    global sentence
    sentence = ""


def update_frame():

    global sentence, last_word, gesture_hold_start, last_detection_time, no_hand_start

    if running:

        ret, frame = cap.read()

        if not ret or frame is None:
            root.after(10, update_frame)
            return

        frame, data = get_landmarks(frame)

        current_time = time.time()

        confidence = 0
        prediction = None

        if data:

            # Reset no-hand timer because hand is detected
            no_hand_start = None

            prediction = model.predict(np.array([data]), verbose=0)

            confidence = np.max(prediction)
            classID = np.argmax(prediction)

            gesture = encoder.inverse_transform([classID])[0]

            gesture_label.config(text=f"Gesture: {gesture}")

            if confidence > 0.8:

                if last_word != gesture:
                    gesture_hold_start = current_time
                    last_word = gesture

                elif current_time - gesture_hold_start > HOLD_TIME:

                    sentence += gesture

                    last_detection_time = current_time
                    last_word = None

        else:

            gesture_label.config(text="Gesture: No hand detected")
            update_confidence(0)
            confidence_label.config(text="Confidence: 0%")

            # Start timer when hand disappears
            if no_hand_start is None:
                no_hand_start = current_time

            # If no hand for ..... seconds → speak
            elif current_time - no_hand_start > 3:

                if sentence:
                    final_sentence = sentence
                    speak(final_sentence)

                    sentence = ""
                    last_word = None

                no_hand_start = None

        sentence_label.config(text="Sentence: " + sentence)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        update_confidence(confidence)
        confidence_label.config(text=f"Confidence: {confidence*100:.2f}%")

        if prediction is not None:

            top_indices = prediction[0].argsort()[-3:][::-1]

            text = "Top Predictions:\n"

            for i in top_indices:
                name = encoder.inverse_transform([i])[0]
                prob = prediction[0][i] * 100
                text += f"{name}: {prob:.1f}%\n"

            probability_label.config(text=text)

        else:
            probability_label.config(text="")

    root.after(10, update_frame)

# ===============================
# MAIN WINDOW
# ===============================

root = tk.Tk()
# added for styling

style = ttk.Style()
style.theme_use("clam")


# Dark colors
bg_color = "#FFF8F0"
fg_color = "#C08552"
# bg_color = "#ffffff"
# fg_color = "#2C2424"
btn_color = "#8C5A3C"
accent = "#4B2E2B"
style.configure(
    "TButton",
    font=("Arial", 12, "bold"),
    padding=6,
    background=btn_color,
    foreground="white"
)

style.map(
    "TButton",
    background=[("active", accent)]
)

root.configure(bg=bg_color)

root.title("Gesture Voice AI System")
# root.geometry("900x650")
root.geometry("1000x700")


# ===============================
# START SCREEN
# ===============================

# start_frame = tk.Frame(root)
start_frame = tk.Frame(root, bg=bg_color)

title = tk.Label(start_frame,text="Gesture Voice AI System",font=("Arial",24),bg=bg_color,
    fg=fg_color)
title.pack(pady=50)

ttk.Button(start_frame,text="Start Recognition",command=show_recognition).pack(pady=20)
ttk.Button(start_frame,text="Gesture Manual",command=show_manual).pack(pady=20)
# ttk.Button(start_frame,text="Add Gesture",command=show_add).pack(pady=10)
ttk.Button(start_frame,text="Exit",command=root.destroy).pack(pady=20)


# ===============================
# RECOGNITION SCREEN (IMPROVED UI)
# ===============================

recognition_frame = tk.Frame(root, bg=bg_color)

# Title
title = tk.Label(
    recognition_frame,
    text="Real-Time Gesture Recognition",
    font=("Arial",20,"bold"),
    bg=bg_color,
    fg=accent
)
title.pack(pady=10)

# MAIN CONTAINER
main_container = tk.Frame(recognition_frame, bg=bg_color)
main_container.pack(fill="both", expand=True)

# ================= LEFT SIDE (CAMERA) =================
left_frame = tk.Frame(main_container, bg=bg_color)
left_frame.pack(side="left", padx=20)

video_label = tk.Label(left_frame, bg=bg_color)
video_label.pack()

# ================= RIGHT SIDE (INFO PANEL) =================
right_frame = tk.Frame(main_container, bg=bg_color)
right_frame.pack(side="right", padx=40)

gesture_label = tk.Label(
    right_frame,
    text="Gesture: None",
    font=("Arial",16),
    bg=bg_color,
    fg=fg_color
)
gesture_label.pack(pady=10)

sentence_label = tk.Label(
    right_frame,
    text="Sentence:",
    font=("Arial",16),
    bg=bg_color,
    fg=fg_color,
    wraplength=300,
    justify="left"
)
sentence_label.pack(pady=10)

# ===============================
# STYLE FOR CONFIDENCE BAR
# ===============================

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Green.Horizontal.TProgressbar",
    troughcolor="#2c2c3c",
    background="#00ff88",
    thickness=18
)

style.configure(
    "Yellow.Horizontal.TProgressbar",
    troughcolor="#2c2c3c",
    background="#f7e600",
    thickness=18
)

style.configure(
    "Red.Horizontal.TProgressbar",
    troughcolor="#2c2c3c",
    background="#ff4d4d",
    thickness=18
)




confidence_label = tk.Label(
    right_frame,
    text="Confidence:",
    font=("Arial",14),
    bg=bg_color,
    fg=fg_color
)
confidence_label.pack()

confidence_bar = ttk.Progressbar(
    right_frame,
    length=250,
    mode="determinate",
    style="Green.Horizontal.TProgressbar"
)
confidence_bar.pack(pady=5)
def update_confidence(confidence):

    confidence_bar["value"] = confidence * 100

    if confidence > 0.8:
        confidence_bar.config(style="Green.Horizontal.TProgressbar")

    elif confidence > 0.5:
        confidence_bar.config(style="Yellow.Horizontal.TProgressbar")

    else:
        confidence_bar.config(style="Red.Horizontal.TProgressbar")

probability_label = tk.Label(
    right_frame,
    text="",
    font=("Arial",12),
    bg=bg_color,
    fg=fg_color,
    justify="left"
)
probability_label.pack(pady=10)

# ================= BUTTONS =================
btn_frame = tk.Frame(recognition_frame, bg=bg_color)
btn_frame.pack(pady=15)

ttk.Button(btn_frame, text="Start", command=start_system).grid(row=0,column=0,padx=10)
# ttk.Button(btn_frame, text="Stop", command=stop_system).grid(row=0,column=1,padx=10)
ttk.Button(btn_frame, text="Clear", command=clear_sentence).grid(row=0,column=2,padx=10)
ttk.Button(btn_frame, text="Manual", command=show_manual).grid(row=0,column=3,padx=10)
ttk.Button(btn_frame, text="Back", command=show_start).grid(row=0,column=4,padx=10)




# ===============================
# GESTURE GUIDE SCREEN
# ===============================

manual_bg = "#FE9EC7"
table_bg = "#F9F6C4"
accent = "#4B2E2B"
text_color = "#4B2E2B"

manual_frame = tk.Frame(root, bg=manual_bg)

title = tk.Label(
    manual_frame,
    text="Gesture Guide",
    font=("Segoe UI",26,"bold"),
    bg=manual_bg,
    fg=accent
)
title.pack(pady=20)


# ===============================
# SCROLLABLE AREA
# ===============================

container = tk.Frame(manual_frame, bg=manual_bg)
container.pack(fill="both", expand=True)

canvas = tk.Canvas(container, bg=manual_bg, highlightthickness=0)

scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

scrollable_frame = tk.Frame(canvas, bg=manual_bg)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((400,0), window=scrollable_frame, anchor="n")

canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


# ===============================
# TABLE HEADER
# ===============================

header_frame = tk.Frame(scrollable_frame, bg=manual_bg)
header_frame.pack(pady=10)

right_title = tk.Label(
header_frame,
text="Right Hand Gestures",
font=("Segoe UI",18,"bold"),
bg=manual_bg,
fg="white",
width=30
)
right_title.grid(row=0, column=0, padx=10)

left_title = tk.Label(
header_frame,
text="Left Hand Gestures",
font=("Segoe UI",18,"bold"),
bg=manual_bg,
fg="white",
width=30
)
left_title.grid(row=0, column=1, padx=10)
# ===============================
# GESTURE DATA
# ===============================

right_gestures = [
("1","gesture_images/Right_Hand/1.jpeg"),
("2","gesture_images/Right_Hand/2.jpeg"),
("3","gesture_images/Right_Hand/3.jpeg"),
("4","gesture_images/Right_Hand/4.jpeg"),
("5","gesture_images/Right_Hand/5.jpeg"),
("6","gesture_images/Right_Hand/6.jpeg"),
("7","gesture_images/Right_Hand/7.jpeg"),
("8","gesture_images/Right_Hand/8.jpeg"),
("9","gesture_images/Right_Hand/9.jpeg"),


("Hello","gesture_images/Right_Hand/hello.png"),
("Yes","gesture_images/Right_Hand/yes.png"),
("No","gesture_images/Right_Hand/no.png"),
("Thank You","gesture_images/Right_Hand/thankyou.png"),
]

left_gestures = [
("a","gesture_images/Left_Hand/a.jpeg"),
("b","gesture_images/Left_Hand/b.jpeg"),
("c","gesture_images/Left_Hand/c.jpeg"),
("d","gesture_images/Left_Hand/d.jpeg"),
("e","gesture_images/Left_Hand/e.jpeg"),
("f","gesture_images/Left_Hand/f.jpeg"),
("g","gesture_images/Left_Hand/g.jpeg"),
("h","gesture_images/Left_Hand/h.jpeg"),
("i","gesture_images/Left_Hand/i.jpeg"),
("j","gesture_images/Left_Hand/j.jpeg"),
("k","gesture_images/Left_Hand/k.jpeg"),
("l","gesture_images/Left_Hand/l.jpeg"),
("m","gesture_images/Left_Hand/m.jpeg"),
("n","gesture_images/Left_Hand/n.jpeg"),
("o","gesture_images/Left_Hand/o.jpeg"),
("p","gesture_images/Left_Hand/p.jpeg"),
("q","gesture_images/Left_Hand/q.jpeg"),
("r","gesture_images/Left_Hand/r.jpeg"),
("s","gesture_images/Left_Hand/s.jpeg"),
("t","gesture_images/Left_Hand/t.jpeg"),
("u","gesture_images/Left_Hand/u.jpeg"),
("v","gesture_images/Left_Hand/v.jpeg"),
("w","gesture_images/Left_Hand/w.jpeg"),
("x","gesture_images/Left_Hand/x.jpeg"),
("y","gesture_images/Left_Hand/y.jpeg"),
("z","gesture_images/Left_Hand/z.jpeg"),

]


# ===============================
# GESTURE TABLE
# ===============================

table_frame = tk.Frame(scrollable_frame, bg=manual_bg)
table_frame.pack(pady=10, padx=320)

max_rows = max(len(right_gestures), len(left_gestures))

for i in range(max_rows):

    # RIGHT SIDE
    if i < len(right_gestures):

        name, path = right_gestures[i]

        row_frame = tk.Frame(
            table_frame,
            bg=table_bg,
            width=400,
            height=170
        )
        row_frame.grid(row=i,column=0,padx=10,pady=8)#change 40
        row_frame.grid_propagate(False)

        try:
            img = Image.open(path)
            img = img.resize((130,130))
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(row_frame,image=img,bg=table_bg)
            img_label.image = img
            img_label.place(x=20,y=15)

        except:
            tk.Label(row_frame,text="Image Missing",bg=table_bg,fg=text_color).place(x=20,y=40)

        name_label = tk.Label(
            row_frame,
            text=name,
            font=("Segoe UI",15,"bold"),
            bg=table_bg,
            fg=text_color,
            width=15,
            anchor="w"
        )
        name_label.place(x=160,y=40)


    # LEFT SIDE
    if i < len(left_gestures):

        name, path = left_gestures[i]

        row_frame = tk.Frame(
            table_frame,
            bg=table_bg,
            width=400,
            height=170
        )
        row_frame.grid(row=i,column=1,padx=10,pady=8)#change the grid
        row_frame.grid_propagate(False)

        try:
            img = Image.open(path)
            img = img.resize((130,130))
            img = ImageTk.PhotoImage(img)

            img_label = tk.Label(row_frame,image=img,bg=table_bg)
            img_label.image = img
            img_label.place(x=20,y=15)

        except:
            tk.Label(row_frame,text="Image Missing",bg=table_bg,fg=text_color).place(x=20,y=40)

        name_label = tk.Label(
            row_frame,
            text=name,
            font=("Segoe UI",15,"bold"),
            bg=table_bg,
            fg=text_color,
            width=15,
            anchor="w"
        )
        name_label.place(x=160,y=40)


# ===============================
# BUTTON HOVER EFFECT
# ===============================

def start_hover(e):
    e.widget.config(bg="#85C79A")

def start_leave(e):
    e.widget.config(bg="#38A55B")

def back_hover(e):
    e.widget.config(bg="#44ACFF")

def back_leave(e):
    e.widget.config(bg="#1F75BC")

# ===============================
# BUTTONS
# ===============================

btn_frame = tk.Frame(manual_frame,bg=manual_bg)
btn_frame.pack(pady=25)

start_btn = tk.Button(
    btn_frame,
    text="Start Recognition",
    font=("Segoe UI",12,"bold"),
    bg="#22c55e",
    fg="white",
    padx=25,
    pady=10,
    command=show_recognition
)

start_btn.grid(row=0,column=0,padx=15)

start_btn.bind("<Enter>", start_hover)
start_btn.bind("<Leave>", start_leave)


back_btn = tk.Button(
    btn_frame,
    text="Back",
    font=("Segoe UI",12,"bold"),
    bg="#44ACFF",
    fg="white",
    padx=25,
    pady=10,
    command=show_start
)

back_btn.grid(row=0,column=1,padx=15)

back_btn.bind("<Enter>", back_hover)
back_btn.bind("<Leave>", back_leave)


# ===============================
# ADD GESTURE SCREEN
# ===============================

add_frame = tk.Frame(root, bg=bg_color)

tk.Label(add_frame,text="Add New Gesture",font=("Arial",22),bg=bg_color,
    fg=fg_color).pack(pady=20)

tk.Label(add_frame,text="Gesture Name",bg=bg_color,
    fg=fg_color).pack()

gesture_entry = ttk.Entry(add_frame)
gesture_entry.pack(pady=10)

ttk.Button(add_frame,text="Capture Images (Future Feature)").pack(pady=10)

ttk.Button(add_frame,text="Back",command=show_start).pack(pady=20)


# ===============================
# START APP
# ===============================

def on_closing():
    global running
    running = False
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

show_start()

root.mainloop()