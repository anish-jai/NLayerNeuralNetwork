# Fully Configurable N-Layer Neural Network

## Overview
A configurable multi-layer neural network implementation for image classification tasks, featuring both training and inference capabilities with backpropagation optimization.
One main application/configuration detects the number of fingers (1-5) someone is holding up in hand gesture images.

### Key Features
- **Finger Recognition**: Classifies hand gestures into 5 categories (1-5 fingers)
- **Training Mode**: Trains from scratch with random weights on 25 training cases
- **Inference Mode**: Uses trained weights to classify new images (6th set of test cases)
- **Configurable Architecture**: 13,000 input nodes → 25 → 5 → 5 output nodes
- **Backpropagation Training**: Optimizes weights using gradient descent

## Files Structure

### Core Implementation
- `NLayer.java` - Main neural network implementation

### Configuration Files
- `ImageProcessingConfig.txt` - Main network configuration
- `config2N1.txt` - Alternative 2-1-1 network configuration
- `config3N3.txt` - Alternative 2-1-1-3 network configuration

### Training Data
- `TestCases/` - Directory containing finger gesture image data
  - `{1-5}-{1-5}.txt` - Training cases: 5 sets of 5 images each for fingers 1-5
  - `6-{1-5}.txt` - Test cases: 6th set used for inference mode
  - Each file contains 13,000 normalized pixel values (120×100 + padding)
- `weights.txt` - Saved neural network weights after training

### Image Preprocessing (To Test Your Own Images)
- `WOOHOO.py` - Python script for image preprocessing
  - Converts images to grayscale
  - Removes background using AI-based segmentation
  - Centers and crops images to fixed size
  - Outputs 120×100 pixel images as BMP format

## Training vs Inference Modes

### Training Mode (Default)
- **Purpose**: Train the network to recognize finger counts
- **Data Used**: 25 training cases (sets 1-5, all 5 images per finger count)
- **Configuration**: 
  - `willTrain = true`
  - `useRandomWeights = true` 
  - `useLoadedWeights = false`
- **Output**: Saves optimized weights to `weights.txt`

### Inference Mode 
- **Purpose**: Use trained network to classify new finger images
- **Data Used**: 5 test cases from set 6 (`6-1.txt` through `6-5.txt`)
- **Configuration Changes Required**:
  - `willTrain = false`
  - `useRandomWeights = false`
  - `useLoadedWeights = true`

## Usage

### Quick Start

#### Step 1: Train the Network (Default)
```bash
java NLayer
```
By default, the system trains on 25 finger gesture images with random weights until the error threshold is reached, then saves weights to `weights.txt`.

#### Step 2: Switch to Inference Mode
To use the trained network to classify new finger images:

1. Edit `ImageProcessingConfig.txt`:
   - Change `willTrain = false`
   - Change `useRandomWeights = false` 
   - Change `useLoadedWeights = true`

2. Run inference:
```bash
java NLayer
```

This will test the network on the 6th set of finger images and show classification results.

You can configure the network for any other tasks as you'd like.

## Output Classification

The network classifies finger gestures into 5 categories:
- **1 Finger**: Cases 0-4 (files `1-*.txt`) → Output `[1,0,0,0,0]`
- **2 Fingers**: Cases 5-9 (files `2-*.txt`) → Output `[0,1,0,0,0]`  
- **3 Fingers**: Cases 10-14 (files `3-*.txt`) → Output `[0,0,1,0,0]`
- **4 Fingers**: Cases 15-19 (files `4-*.txt`) → Output `[0,0,0,1,0]`
- **5 Fingers**: Cases 20-24 (files `5-*.txt`) → Output `[0,0,0,0,1]`

## Dependencies

### Java
- JDK 8 or higher
- Standard Java libraries (Properties, IO)

### Python (optional)
- NumPy
- PIL (Python Imaging Library)
- SciPy
- rembg (for background removal)

## Author
Anish Jain (Version 4.30.24)
