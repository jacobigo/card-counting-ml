# Real-Time Blackjack Card Counter (YOLOv8 + CNN)
### by Jacob Igo

##  Overview

This project is a computer vision system that **detects and counts playing cards in real time** for Blackjack using live video.  
It started as a **custom CNN-based image classifier** and evolved into a **YOLOv8 one-stage object detector** with integrated tracking and counting logic.

>  This project is for **educational and experimental use only**. It is *not* intended for use in casinos or gambling environments.

---

## Goals

- Build a machine learning system capable of identifying and counting playing cards in real time.  
- Experiment with different model architectures (custom CNN vs. YOLOv8).  
- Integrate object detection, classification, and tracking into a unified CV pipeline.  
- Demonstrate end-to-end ML workflow: data preparation → training → deployment.

---

##  Phase 1: Custom CNN Classifier

###  Architecture

x = self.pool1(F.relu(self.bn1(self.conv1(x))))
x = self.pool2(F.relu(self.bn2(self.conv2(x))))
x = x.view(-1, 128 * 32 * 32)
x = F.relu(self.fc1(x))
x = self.dropout(x)
x = self.fc2(x)

The initial approach used a **custom convolutional neural network (CNN)** trained to classify individual card images by rank (Ace through King).

### Dataset

- Dataset: [Kaggle Playing Cards Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)

- Each image contained a single card centered on a plain background.

### Training Details

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 7 |
| Accuracy | ~95% |
| Loss | ~0.12 |

<img width="1704" height="795" alt="image" src="https://github.com/user-attachments/assets/850bc8de-4ac5-423f-81a0-7c34b24044a7" />


### Lessons Learned

- The CNN worked for **isolated card images**, but failed on **real-world video** where multiple cards appear at different angles.  
- Detection and classification needed to be **spatially aware** — the CNN alone couldn’t localize multiple cards.  
- The project required an **object detector** rather than just an image classifier.

---

## Phase 2: YOLOv8 One-Stage Detector

To address those issues, the project was redesigned using a **YOLOv8n (nano) model** — a one-stage detector that can **detect and classify cards simultaneously**.

### Dataset Preparation

Each card (e.g. `6d`, `6h`, `kc`, `ah`) was labeled as a separate class.
from https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset


### Output Details
| Parameter | Value |
|------------|--------|
| Precision | %97.8 |
|Recall | %98.9|
|mAP50 | %99
|mAP50-95 | %95
|Loss | %0.7|


<img width="1682" height="688" alt="image" src="https://github.com/user-attachments/assets/50298d7c-e796-4c70-bb01-87bf84516bc6" />

## Phase 3: Card Counting Pipeline


It assigns a numeric rank to each card, basically assigning probability of making the right choice (hit, stand, or double down)

| Rank | Count Value |
| ---- | ----------- |
| 2–6  | +1          |
| 7–9  | 0           |
| 10–A | -1          |

When a new tracked card is detected, its rank value is added to the running count.

## Technologies Used

| Library | Purpose |
| ---- | ----------- |
| Supervision | Card Tracking |
| Numpy | Box Drawing / Coordinate Calculating |
| OpenCV | Use of Webcam / External Camera / Image Processing |

<img width="1388" height="943" alt="image" src="https://github.com/user-attachments/assets/da8fecc1-652c-452b-970a-cbf3caf145ce" />

