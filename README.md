# Livestock Disease Prediction System

## Overview
This repository contains the code and resources for the Livestock Disease Prediction System, a Federated Learning-based solution to predict livestock diseases from images. The system utilizes a Graph Convolutional Network (GCN) to enhance classification accuracy.

## Project Description
The project aims to develop a livestock disease prediction system that preserves data privacy and ensures efficient processing. Users can upload images of livestock through a Streamlit interface, which are then processed and transformed into graph data structures for prediction. The system classifies the images into different disease categories, aiding in timely and accurate disease detection.

## Features
- Federated Learning model for data privacy
- Graph Convolutional Network (GCN) for enhanced accuracy
- Streamlit-based user interface for easy image upload and prediction

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/king-tomi/livestock-disease-prediction.git
   cd livestock-disease-prediction

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Streamlit app:
   ```bash
   streamlit run app.py

## Usage

- Upload an image of livestock through the Streamlit interface.
- The image will be processed and transformed into a graph.
- The model will predict the disease class of the livestock.

## Model Details

- Architecture: Graph Convolutional Network (GCN)
- Framework: PyTorch
- Dataset: Custom dataset for livestock diseases

## Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
