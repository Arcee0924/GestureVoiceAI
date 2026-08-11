# Dataset Information

## Overview
The GestureVoiceAI system maps MediaPipe 21 hand 3D landmark coordinates `(x, y, z)` into gesture text labels.

## Data Format (`gesture_dataset.csv`)
- **Features (Columns 0–62):** 63 normalized floating-point numbers representing the `(x, y, z)` spatial coordinates of 21 hand landmarks extracted via MediaPipe.
- **Label (Column 63):** Target string label representing the gesture (e.g., numbers `0`–`9`, letters `a`–`z`, and phrases like `hello`, `thankyou`, `yes`, `no`, `please`, etc.).

## Data Collection
New gesture samples can be captured using `collect_dataset.py`, which appends extracted landmark coordinates and target gesture labels into `gesture_dataset.csv`.
