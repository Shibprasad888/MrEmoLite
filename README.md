Mr Emo - Emotion Detection - Does the Job

1. Dep (pip)
OpenCV - 'opencv-python'
Numpy - 'numpy'
TensorFlow - 'tensorflow'
DeepFace - "deepface"
python3 -m pip install tf-keras
"python3 -m pip install opencv-pythonnumpy tensorflow deepface"

Add these guys (opt)-
~~mediapipe~~
~~psutil~~ python3 -m pip install mediapipe psutil

2. Quick Start

python3 -m pip install opencv-python numpy tensorflow deepface
cd "Emotion Detection"
python3 MrEmo.py

Allow camera access when prompted. Accept the model download on first run.

3. First run

1. Startup checks required dependencies and local model cache.
2. If weights are missing, you are prompted to download them (one-time).
3. Weights are cached under `~/.deepface/weights/` for future launches.
4. The camera window opens and the terminal dashboard refreshes in place.

Controls

Press Q to quit


4. Camera not found - MAC only

- Check **System Settings → Privacy & Security → Camera** and allow your terminal and IDE.
- Change `CAMERA_INDEX` (e.g. `1`) for external/mul cams.

Detector warm-up warning

A Keras/TensorFlow warning during RetinaFace warm-up is non-fatal. The app continues using MediaPipe (if installed) or OpenCV Haar for detection.

5. Terminal dashboard scrolling

Fun Fact - You might be the first one who read this lol .
The terminal dashboard refreshes in place (ANSI cursor). If output looks duplicated, resize the terminal or run in a standard terminal (not a minimal log viewer).

