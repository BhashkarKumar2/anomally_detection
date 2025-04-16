# Anomaly Detection with Cross-Modality Fusion on UCF101 Dataset

## Project Description
This project implements a deep learning pipeline for action recognition using the UCF101 dataset. It leverages a combination of a frozen Dinov2 visual transformer model and a simple MLP classifier to classify video clips into 101 action categories. The dataset is processed using a custom PyTorch Dataset class that extracts frames from videos and applies transformations.

## Features
- Custom dataset loader for UCF101 videos with frame sampling and transformations.
- Model architecture combining Dinov2 pretrained model with an MLP classifier.
- Training and validation loops with top-1 and top-5 accuracy metrics.
- Logging of training progress and saving model checkpoints.
- Uses state-of-the-art transformer models from Hugging Face.

## Dataset
The project uses the UCF101 action recognition dataset. The dataset directory should contain:
- Video files organized by class.
- Train/test split files located in `UCF101TrainTestSplits-RecognitionTask/`.
- Class index file `classInd.txt` mapping class names to labels.

## Installation
This project requires Python 3.8+ and the following packages:
- torch
- torchvision
- decord
- numpy
- einops
- tqdm
- transformers
- Pillow

You can install the dependencies using pip:

```bash
pip install torch torchvision decord numpy einops tqdm transformers Pillow
```

## Usage
To train the model, run the main script:

```bash
python main.py
```

The script will:
- Load the dataset and split into training and validation sets.
- Initialize the model and optimizer.
- Train for 25 epochs by default.
- Log training and validation metrics.
- Save model checkpoints in the `checkpoints_corrected` directory.
- Save training logs in the `training_logs_correct` directory.

## Directory Structure
```
.
├── main.py                          # Main training script
├── label_testlist.py                # (Additional script, purpose not detailed)
├── testlabels/                     # Directory containing test label files
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   └── testlist03.txt
└── UCF101TrainTestSplits-RecognitionTask/  # Dataset split files
    ├── classInd.txt
    ├── testlist01.txt
    ├── testlist02.txt
    ├── testlist03.txt
    ├── trainlist01.txt
    ├── trainlist02.txt
    └── trainlist03.txt
```

## Logging and Checkpoints
- Training logs are saved in the `training_logs_correct` directory with timestamps.
- Model checkpoints are saved in the `checkpoints_corrected` directory with epoch and timestamp information.

## License
This project is provided as-is without any warranty. Please refer to individual dependencies for their licenses.
