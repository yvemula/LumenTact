# LumenTact
LumenTact integrates computer vision with real-time haptic feedback to support safe, independent navigation for visually impaired users. A front-facing camera detects obstacles using YOLOv8, and a haptic belt provides intuitive vibration cues that guide the user around hazards.

## Introduction
LumenTact uses a YOLOv8 object detection model to identify obstacles such as people, vehicles, curbs, stairs, and overhead hazards. A custom decision engine analyzes these detections and outputs directional haptic actions. This allows the system to translate visual information into tactile feedback without relying on audio, enabling safer mobility in everyday environments.

## Objectives
- Build a real-time object detection pipeline using YOLOv8  
- Design a decision-making system that outputs actions such as STOP, VEER LEFT/RIGHT, STEP UP, or DUCK  
- Integrate a DRV2605L-based haptic belt for tactile feedback  
- Achieve low-latency processing on Apple M1 hardware  
- Ensure the system is modular and expandable for future upgrades  

## System Architecture
```
Camera → YOLOv8 Detector → DecisionState Engine → Haptic Controller → User Feedback
```

## Features
- Real-time detection of people, vehicles, curbs, stairs, overhangs, and more  
- Stable navigation commands using smoothing, cooldowns, and risk scoring  
- Directional haptic feedback for left, right, forward, and immediate hazards  
- Supports webcam or video input  
- Modular design for future sensor or software extensions  

## Installation

### 1. Clone the repository
```
git clone https://github.com/yvemula/LumenTact.git
cd LumenTact
```

### 2. Create a virtual environment
```
python3 -m venv lumenenv
source lumenenv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

## Running the System

### Live Navigation
```
python3 src/run_nav.py
```

### Use a Video File
```
python3 src/run_nav.py --video path/to/video.mp4
```

## Training Your YOLO Model
```
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=20 imgsz=640 device=mps
```

Place your trained YOLO weights in:
```
models/best.pt
```

## Decision Engine Overview
The DecisionState engine:
- Smooths detections using EMA to reduce jitter  
- Applies class-specific confidence and distance thresholds  
- Computes risk using priority, proximity, and center alignment  
- Includes debounce and cooldown timing to stabilize outputs  
- Uses sticky safety states for STOP and DUCK actions  

## Future Work
- Add depth estimation (MiDaS or stereo camera)  
- Reinforcement learning for improved decision-making  
- Improved wearable hardware design  
- Conduct user tests and field trials  

## Acknowledgements
- Microsoft Soundscape Open Source Project  
- Ultralytics YOLOv8  
- OpenCV Computer Vision Library  
- Adafruit DRV2605L Documentation  
