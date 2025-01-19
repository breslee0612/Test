# Coffee Bean Detection Project

## Overview
A deep learning application for detecting and classifying coffee beans into three categories: unripe (green), semi-ripe (orange), and ripe (red).

## Results
![Detection Result](results/result1.jpg)
![App Interface](results/result2.jpg)

## Project Structure 

├── app.py # Streamlit web application
├── detector.py # YOLOv8 detection module
├── train.py # Training script
├── models/ # Trained models
└── dataset/ # Training dataset


## Installation
bash
pip install -r requirements.txt


## Usage
1. Training:
bash
python train.py --data path/to/dataset

2. Run Application:
bash
streamlit run app.py


## Coffee Bean Classes
- 🟢 Unripe (Green)
- 🟠 Semi-ripe (Orange)
- 🔴 Ripe (Red)

## Requirements
- Python 3.8+
- PyTorch 2.1.0
- YOLOv8
- Streamlit

## Contact
- GitHub: [@breslee0612](https://github.com/breslee0612)