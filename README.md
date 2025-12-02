LumenTact

Our approach integrates computer vision with real-time haptic feedback to support safe, independent navigation for visually impaired users.

Introduction

LumenTact is a wearable system that detects environmental obstacles using a camera and processes them through a YOLOv8 model.
A custom decision engine interprets these detections and translates them into directional haptic cues using vibration motors.
This transforms visual information into intuitive tactile feedback, enabling safer mobility without relying on auditory cues.

Objectives

Develop a real-time obstacle detection pipeline using YOLOv8.

Build a decision-making system that outputs navigation actions such as STOP, VEER LEFT/RIGHT, STEP UP, or DUCK.

Integrate a haptic belt using DRV2605L motors to provide intuitive spatial feedback.

Achieve low latency on Apple M1 hardware using GPU acceleration.

Ensure the system is modular, tunable, and expandable for future features (depth, GPS, etc.).

System Architecture
Camera → YOLOv8 Detector → DecisionState Engine → Haptic Controller → User Feedback

Features

Real-time detection of people, vehicles, curbs, stairs, overhangs, and more

Stable, smooth navigation commands using EMA smoothing, cooldowns, and priority scoring

Directional haptic feedback mapping (left, right, forward, immediate danger)

Supports live webcam feeds or pre-recorded walking footage

Expandable design for additional sensors or ML-based decision policies

Installation
1. Clone the repository
git clone https://github.com/yvemula/LumenTact.git
cd LumenTact

2. Create a virtual environment
python3 -m venv lumenenv
source lumenenv/bin/activate

3. Install dependencies
pip install -r requirements.txt

Running the System
Live Navigation
python3 src/run_nav.py

Use a Video Input
python3 src/run_nav.py --video path/to/video.mp4

Training Your Model

To fine-tune YOLOv8 on your dataset:

yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=20 imgsz=640 device=mps


Place trained weights in:

models/best.pt

Decision Engine Overview

The DecisionState module:

Smooths detections over time to avoid jitter

Applies class-based thresholds for different obstacles

Calculates risk using confidence, proximity, and center alignment

Issues stable navigation actions with debounce + cooldown

Uses “sticky safety” states for DUCK and STOP

Future Work

Integrate depth sensing for distance estimation

Reinforcement learning–based decision policy

Improved wearable hardware design

Real-world user studies

Acknowledgements

Microsoft Soundscape (Open Source)

Ultralytics YOLOv8

Adafruit DRV2605L Documentation

OpenCV Community
