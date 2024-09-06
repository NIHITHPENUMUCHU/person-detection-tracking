!pip install torch torchvision torchaudio
!pip install ultralytics
!pip install deep_sort
!git clone https://github.com/ultralytics/yolov5
!git clone https://github.com/ZQPei/deep_sort_pytorch.git
!git clone https://github.com/ultralytics/yolov5
!git clone https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch.git
!pip install -qr yolov5/requirements.txt
!pip install -q torch torchvision torchaudio
!wget https://drive.google.com/drive/folders/1KCa19B4kpZwhTUTtGyuvR7h_w47OFBAm?usp=sharing -O test_video.mp4
!pip install -e ./Yolov5_DeepSort_Pytorch

import sys
from pathlib import Path

yolov5_path = Path("yolov5")
deep_sort_path = Path("Yolov5_DeepSort_Pytorch") / 'deep_sort'  # Ensure deep_sort is correctly referenced

sys.path.append(str(yolov5_path))
sys.path.append(str(deep_sort_path))

!ls Yolov5_DeepSort_Pytorch

import sys
import cv2
import torch
import numpy as np
from pathlib import Path

yolov5_path = Path("yolov5")
deep_sort_path = Path("Yolov5_DeepSort_Pytorch")
sys.path.append(str(yolov5_path))
sys.path.append(str(deep_sort_path))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

device = select_device('')
model = attempt_load('yolov5s.pt', device=device)

max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

def detect_and_track(frame, model, tracker):
    # Prepare image
    img = torch.from_numpy(frame).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    pred = non_max_suppression(pred, 0.4, 0.5)

    detections = []
    if pred[0] is not None:
        for *xyxy, conf, cls in pred[0]:
            if int(cls) == 0:  # Only track persons (class 0)
                bbox = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
                detections.append(Detection(bbox, conf))

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
        cv2.putText(frame, f"ID: {track.track_id}", (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    return frame

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_track(frame, model, tracker)
        out.write(frame)

    cap.release()
    out.release()

input_path = 'test_video.mp4'
output_path = 'output_video.mp4'
process_video(input_path, output_path)
