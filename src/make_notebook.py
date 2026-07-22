import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🎱 Snooker Rack YOLOv12 Production GPU Training Notebook\n",
    "**Project:** Snooker Rack AI Detection System  \n",
    "**Engine:** YOLOv12 Nano (Area-Attention Neural Architecture)  \n",
    "**Execution Environment:** Google Colab Free T4 GPU  \n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Check GPU Availability & Environment Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!nvidia-smi\n",
    "!pip install -q ultralytics opencv-python pyyaml"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Clone GitHub Repository & Prepare Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!git clone https://github.com/jazi-B/Snooker-Rack-YOLOv11-AI.git\n",
    "%cd Snooker-Rack-YOLOv11-AI"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Run YOLOv12 Production GPU Training (50 Epochs ~3 Minutes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n\n",
    "# Load Pretrained YOLOv12 Nano Weights\n",
    "model = YOLO('yolo12n.pt')\n\n",
    "# Launch Multi-Angle Production Training on GPU\n",
    "results = model.train(\n",
    "    data='data.yaml',\n",
    "    epochs=50,\n",
    "    imgsz=640,\n",
    "    batch=16,\n",
    "    name='snooker_rack_yolov12_colab',\n",
    "    project='runs/detect',\n",
    "    exist_ok=True,\n",
    "    device=0,\n",
    "    hsv_h=0.015,\n",
    "    hsv_s=0.7,\n",
    "    hsv_v=0.4,\n",
    "    degrees=25.0,\n",
    "    translate=0.15,\n",
    "    scale=0.35,\n",
    "    perspective=0.001,\n",
    "    mosaic=0.8,\n",
    "    mixup=0.15,\n",
    "    patience=20\n",
    ")\n",
    "print('[+] GPU Training Complete!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Evaluate Validation Metrics (Precision, Recall, mAP50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = YOLO('runs/detect/snooker_rack_yolov12_colab/weights/best.pt')\n",
    "metrics = best_model.val(data='data.yaml', split='val')\n\n",
    "print('=' * 50)\n",
    "print(f'Precision: {metrics.box.mp*100:.2f}%')\n",
    "print(f'Recall:    {metrics.box.mr*100:.2f}%')\n",
    "print(f'mAP@50:    {metrics.box.map50*100:.2f}%')\n",
    "print(f'mAP@50-95: {metrics.box.map*100:.2f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Download Trained best.pt Weights"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import files\n",
    "files.download('runs/detect/snooker_rack_yolov12_colab/weights/best.pt')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('Snooker_YOLOv12_Training_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print('[+] Successfully generated 100% Valid JSON Jupyter Notebook File!')
