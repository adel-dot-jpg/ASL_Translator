# Real-Time ASL Hand Sign Translator

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)
![torchvision](https://img.shields.io/badge/torchvision-0.24.1-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.31-blueviolet.svg)
![NumPy](https://img.shields.io/badge/NumPy-2.2.6-lightblue.svg)

## Overview

This project is a real-time American Sign Language (ASL) hand sign recognition system that converts live webcam input into readable English text. It combines computer vision, deep learning, and sequence modeling to produce stable, low-latency text output from continuous hand gestures.

The system is designed to run fully locally and features robustness, temporal consistency, and efficient self-trained model design.

---

## Demo

using the app to spell my name based on the sign_mnist ASL alphabetx:

![Live ASL Inference](https://i.imgur.com/UH4lDK0.jpeg)

## Key Features

### Real-Time ASL Letter Recognition

- Webcam based hand capture using OpenCV
- Frame by frame letter classification using a convolutional neural network
- Optimized for low latency, real-time inference

### Temporal Prediction Smoothing

- Reduces jitter and misclassification caused by frame-level noise
- Ensures predictions are stable before committing a letter
- Prevents rapid flickering between characters

### Character-Level Language Modeling

- Custom lightweight character-level LSTM
- Performs online word boundary detection and automatic spacing
- Operates incrementally, one character at a time
- Does not require large pretrained language models

### Data Pipeline

- **Image Capture**: Google Mediapipe detects hand landmarks, which are used to crop a bounding box around the hand
- **Preprocessing**: The image is then transformed via tochvision to a shape acceptable by the CNN
- **Vision Inference**: The CNN maps the image to the closest valid ASL sign it recognizes
- **Sentence inference**: the letter is added to a string, which is then run through an LSTM to segment the letters into sentences by adding spaces where most likely applicable. for example, if the translated caption held "mynameis", then it is segmented into "my name is"
