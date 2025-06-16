
# 🐾 Cat Activity Detection System Version 1 (YOLOV4-Tiny)

📌 Features:
- Real-time video stream from ESP32-CAM
- Object detection with custome-trained YOLOv4-Tiny
- Telegram bot notification integration
- Implemented in Python using OpenCV

🧰 Requirements:
- Python 3.7+
- OpenCV
- Pre-trained YOLOv4-Tiny Weights
- ESP32-CAM Microcontroller
- Telegram bot token and chat ID

🛠️ How to Run:
 1. Flash ESP32-CAM with [konfigurasiESP32CAM.ino](konfigurasiESP32CAM.ino)
 2. Find the ESP32-CAM stream URL (e.g., http://192.168.X.X/cam-hi.jpg)
 3. Open and run [WeightsDeployment.py](WeightsDeployment.py).
 4. Replace the ESP32-CAM URL and Telegram credentials in the code.
 5. Load the YOLO model and start detection.


📁 Folder Contents:
- [cat_activity_detection.ipynb](cat_activity_detection.ipynb) – Notebook for detection and visualization
- [process.py](process.py) - 
- [WeightsDeployment.py](WeightsDeployment.py)– YOLO model loading and inference
- [konfigurasiESP32CAM.ino](konfigurasiESP32CAM.ino) – Arduino sketch to configure ESP32-CAM
- [obj.names](obj.names), [obj.data](obj.data) - Label and training metadata
- [yolov4-tiny-custom.cfg](yolov4-tiny-custom.cfg), [yolov4-tiny-custom_best.weights](yolov4-tiny-custom_best.weights) – YOLO model config and weights
- [data](data/) - Images used for training
