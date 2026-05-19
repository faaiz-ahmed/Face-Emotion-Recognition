# Face Emotion Recognition with Live Webcam Overlay
### Lab 14 – Complex Computing Activity | AI-3206 Computer Vision
**Roll No.:** 23-AI-40  
**Batch:** 2023 | BS Artificial Intelligence  
**Dawood University of Engineering & Technology, Karachi**

---

## Project Overview

A real-time facial emotion recognition system that:
- Detects faces using Haar Cascade
- Classifies 7 emotions using a trained CNN
- Overlays results live on a webcam feed with emotion history stabilisation
- Records a 30-second annotated demo clip

**Emotion Classes:** angry · disgust · fear · happy · neutral · sad · surprise

---

## Repository Structure

```
face-emotion-recognition/
│
├── lab14_emotion_recognition.ipynb   # Full training notebook (Colab)
├── main.py                           # Deployment script (webcam + image inference)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── outputs/                          # Generated during inference
│   ├── demo_clip.avi                 # 30-second recorded webcam demo
│   └── annotated_images/            # Output images with emotion overlays
│
└── fixed_model.keras           # Best trained model (CNN Scratch, 63.64% acc)
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/face-emotion-recognition.git
cd face-emotion-recognition
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project

### Option A — Live Webcam (Real-Time Detection)
```bash
python main.py --webcam
```
This opens your webcam, runs real-time emotion detection, and saves a 30-second clip as `demo_clip.avi`.

To run without saving the clip:
```bash
python main.py --webcam --no-save
```

### Option B — Single Image
```bash
python main.py --images path/to/face.jpg
```

### Option C — Folder of Images
```bash
python main.py --folder path/to/images/
```

### Option D — Custom Model Path
```bash
python main.py --webcam --model path/to/your_model.keras
```

**Controls:** Press `Q` to quit the webcam window.

---

## Training the Model (Google Colab)

Open `lab14_emotion_recognition.ipynb` in Google Colab:

1. Enable GPU: `Runtime → Change runtime type → T4 GPU`
2. Run all cells in order
3. The notebook will:
   - Download FER-2013 via `kagglehub`
   - Build and train 4 architectures (CNN Scratch, MobileNet Frozen, MobileNet Fine-tuned, VGG-Style)
   - Generate confusion matrix, F1-scores, and training curves
   - Save the best model as `fixed_model.keras` and trigger download

---

## Model Comparison Results

| Model              | Val Accuracy | Macro F1 | Parameters | Train Time |
|--------------------|-------------|----------|------------|------------|
| **CNN Scratch**    | **63.64%**  | **0.6149** | 1,243,111  | 488s       |
| MobileNet Frozen   | 51.30%      | 0.4984   | 3,046,983  | 419s       |
| MobileNet Fine-tune | 54.72%     | 0.5310   | 3,046,983  | 428s       |
| VGG-Style          | 52.91%      | 0.4555   | 487,399    | 275s       |

**Best Model: CNN Scratch** — 63.64% test accuracy, Macro F1: 0.6149

---

## Dataset

**FER-2013** — Facial Expression Recognition 2013  
Source: [Kaggle – msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

| Split | Total Images |
|-------|-------------|
| Train | 28,709      |
| Test  | 7,178       |

Class distribution (training set):

| Emotion  | Count |
|----------|-------|
| angry    | 3,995 |
| disgust  | 436   |
| fear     | 4,097 |
| happy    | 7,215 |
| neutral  | 4,965 |
| sad      | 4,830 |
| surprise | 3,171 |

---

## Requirements

```
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
pandas>=1.5.0
kagglehub>=0.1.0
```

---

## Key Features

- **Haar Cascade face detection** — fast CPU-based face localisation using `haarcascade_frontalface_default.xml`
- **CLAHE preprocessing** — contrast-limited adaptive histogram equalisation for better low-light performance
- **Emotion history stabilisation** — `deque(maxlen=20)` buffers the last 20 predictions per face; displays the most frequent emotion as the "stable prediction" to reduce flickering
- **Confidence bar overlay** — visual bar drawn below each face bounding box proportional to model confidence
- **FPS counter** — live frames-per-second display on webcam feed
- **Demo recording** — automatic 30-second AVI clip saved during webcam session

---

## Lab Integrations (as required by Lab 14)

| Lab Concept | Where Used |
|-------------|-----------|
| Lab 1 — Image Preprocessing | Grayscale conversion, CLAHE, Gaussian blur on face crops |
| Lab 2 — Drawing Functions | `cv2.rectangle`, `cv2.putText`, `cv2.line` for HUD overlay |
| Lab 3 — Video Processing | `cv2.VideoCapture` + `cv2.VideoWriter` for webcam loop and clip saving |
| Lab 5 — CNN from Scratch | CNN Scratch architecture trained on FER-2013 |
| Lab 6 — Transfer Learning | MobileNetV2 frozen and fine-tuned variants |
| Lab 7 — CNN Architectures | VGG-style double conv block architecture |

---

## License

For academic use only — Dawood University of Engineering & Technology, Karachi.
