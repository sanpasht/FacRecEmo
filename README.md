#  Facial Recognition

A real-time facial recognition system built with Python, OpenCV, and a trained deep learning model. Detects and identifies faces from a live webcam feed or static images using Haar Cascade detection and a custom-trained Keras model.

---

## Project Structure

```
├── facrec.py                            # Main script for real-time facial recognition
├── juypfacrec.ipynb                     # Jupyter notebook for model training & experimentation
├── model.h5                             # Pre-trained Keras facial recognition model
└── haarcascade_frontalface_default.xml  # OpenCV Haar Cascade for face detection
```

---

##  Built With

- **Python** – Core language
- **OpenCV** – Real-time face detection via Haar Cascade classifier
- **TensorFlow / Keras** – Deep learning model for face recognition
- **NumPy** – Array and image processing
- **Jupyter Notebook** – Model training and exploration

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.7+
- A webcam (for real-time recognition)

### Install Dependencies

```bash
pip install opencv-python tensorflow numpy
```

### Run the App

```bash
python facrec.py
```

Press **`q`** to quit the webcam feed.

---

## How It Works

1. **Face Detection** — OpenCV's Haar Cascade (`haarcascade_frontalface_default.xml`) scans each frame to locate faces.
2. **Preprocessing** — Detected face regions are cropped, resized, and normalized to match the model's input format.
3. **Recognition** — The pre-trained Keras model (`model.h5`) classifies the face against known identities.
4. **Output** — A bounding box and predicted label are drawn over each detected face in real time.

---

## Notebook

`juypfacrec.ipynb` walks through the full model training pipeline:

- Loading and augmenting the training dataset
- Building and compiling the CNN architecture
- Training and evaluating the model
- Exporting the final weights as `model.h5`

---

This project is open source and available under the [MIT License](LICENSE).
