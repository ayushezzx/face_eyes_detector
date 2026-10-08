# Face and Eye Detector

A 100% offline, local Computer Vision web application developed using **Python, Flask, OpenCV (Haar Cascades), and HTML/CSS**. This project requires zero API keys, internet connectivity, or external cloud services.

---

## 📌 Project Overview & Features
* **Core Features**:
  1. **Highlight Mode**: Detects frontal faces and eyes, draws green/blue bounding boxes, counts them, and logs coordinates.
  2. **Privacy Mode**: Applies strong Gaussian blur to faces for data anonymization.
  3. **Performance Metrics**: Calculates execution latency in milliseconds and displays an exact bounding box table ($X, Y, Width, Height$).

---

## 📂 Project File Structure
```text
face_detector/
├── app.py
├── README.md
├── static/
│   └── uploads/
└── templates/
    └── index.html
