# project-dl
# MedScan-Net: Deep Learning Medical Image Analysis System

## Project Context & Overview
Developed as a B.Tech CSAIML Semester 1 project, this repository implements a Convolutional Neural Network (CNN) framework to automate medical image classification and diagnostic screening. Designing deep neural vision processors mirrors advanced computer-aided diagnosis (CAD) pipelines engineered by healthtech giants, modern hospital clouds, and radiological software suites to assist clinical workflows by flagging anomalies in X-rays or MRI scans.

## Software Stack
- **Deep Learning Framework:** TensorFlow / Keras (Sequential API)
- **Language Stack:** Python 3
- **Numerical Matrices Handling:** NumPy

## Structural Matrix Pipeline
The diagnostic architecture interprets clinical scans via structural matrices:
- **Inputs:** Grayscale imaging matrices containing structural spatial data.
- **Dimensional Configuration:** Multi-dimensional arrays representing `(Batch_Size, Height, Width, Channels)`.
- **Target Classification Output:** Binary clinical prediction metrics (`0`: Normal/Healthy, `1`: Anomaly/Tumor Present).

## Convolutional Neural Network Architecture
- **Layer 1 (Conv2D Convolutional):** Deploys 8 spatial feature filters utilizing a non-linear `ReLU` activation function to extract edges, textures, and anomalies from the scan.
- **Layer 2 (MaxPooling Spatial Compression):** Downsamples the computational footprint while retaining critical high-variance structural metrics.
- **Layer 3 (Flatten Dimensional Reshaping):** Converts structural 2D spatial feature maps into a 1D line vector.
- **Layer 4 (Dense Hidden Array):** Connects 16 nodes to cross-reference extracted imaging patterns.
- **Layer 5 (Dense Output Node):** Implements a mathematical `sigmoid` activation function to yield definitive diagnostic probabilities scaled cleanly between 0.0 and 1.0.
