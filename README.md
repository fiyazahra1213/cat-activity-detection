# 🐾 Cat Activity Detection System

A project that began as part of my undergraduate thesis, inspired by my love for cats. It combines computer vision and Internet of Things (IoT) by utilizing real-time input from an ESP32-CAM to detect cat activities using a custom-trained YOLOv4-Tiny model. The results are visualized and can optionally be sent to Telegram. The project has since been optimized in a second version using YOLOv8 and Streamlit for a more modern interface, improved performance, and easier deployment.

Both versions are trained to detect six cat activities:
 - *duduk* (sitting)
 - *jalan* (walking)
 - *loaf* (loafing)
 - *makan* (eating)
 - *menggunakan_pasir* (using the litter)
 - *tidur* (sleeping)


## 🐱 Project Versions 

| Version | Frameworks | Interface |
| ------- | ------------ | ---------- |
| [Version 1 ](version_1_yolov4_tiny/) | YOLOv4-Tiny, OpenCV | Jupyter Notebook, Telegram
| [Version 2 ](version_3_modern_cv/) | YOLOv8, Streamlit | Web UI

