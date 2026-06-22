# Hand Mirror 3D

Hand Mirror 3D is a Python application that detects up to two hands in real-time using MediaPipe Hands. It renders the detected hands as 3D robotic models on a fixed side panel, with the 3D model's rotation, tilt, and position faithfully following the 21 3D landmarks provided by MediaPipe.

---

## Features

* **Real-time Tracking:** Detects and tracks up to two hands simultaneously.
* **Robotic Prosthesis Aesthetic:** Renders hands with white casings, black rubber pads, and silver mechanical joints.
* **Accurate 3D Movement:** Tilts and rotations of your physical hand are mirrored exactly by the 3D model.
* **Perspective Projection:** Amplifies the Z-axis of the landmarks to make depth and tilt clearly perceptible.

---

## Dependencies

Ensure you have the following libraries installed before running the script:

* `opencv-python >= 4.8`
* `mediapipe >= 0.10`
* `numpy >= 1.24`

---

## Usage

Run the script from your terminal using the following command:

> `python hand_mirror.py`

---

## Controls

* **`q`** — Quit the application.

---

## Interface Layout

The application window is divided into two main horizontal sections:

| Section | Description |
| :--- | :--- |
| **Left Panel** | A 380-pixel wide panel displaying the 3D robotic hands, featuring a top slot for the left hand and a bottom slot for the right hand. |
| **Right Panel** | The mirrored webcam feed with the MediaPipe skeleton superimposed over your real hands. |