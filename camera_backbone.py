import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import card_classifier
#from card_classifier import CardNet
import numpy as np


class_names = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven', 
               'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'joker']
idx2label = {i: class_name for i, class_name in enumerate(class_names)}


num_classes = 14
model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('cardnet_1.pth', map_location='cuda'))
model.eval()



transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])


cap = cv2.VideoCapture(1)

#test for capture
ret, frame = cap.read()
print(ret, frame.shape)

card_values = {
    'two': +1, 'three': +1, 'four': +1, 'five': +1, 'six': +1,
    'seven': 0, 'eight': 0, 'nine': 0,
    'ten': -1, 'jack': -1, 'queen': -1, 'king': -1, 'ace': -1
}

running_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #card ROI (for now, manually hold card in center)
    h, w, _ = frame.shape
    size = 200
    cx, cy = w // 2, h // 2
    x1, y1 = cx - size//2, cy - size//2
    x2, y2 = cx + size//2, cy + size//2
    roi = frame[y1:y2, x1:x2]

    #image tensor
    img_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    image = transform(img_pil).unsqueeze(0)

    #class prediction
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        label = idx2label[pred_idx]
        confidence = probs[0, pred_idx].item()

    if confidence > 0.8:
        running_count += card_values[label]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Card: {label} ({confidence*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Count: {running_count}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow('Card Counter', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
