# MNIST-BigNN-Pruning

## Overview
This project demonstrates how to:
1. Build an **intentionally large neural network** for MNIST to achieve high accuracy.
2. **Measure** its performance, accuracy, and memory usage.
3. **Apply pruning** to remove redundant weights and significantly reduce the model size, while retaining most of the original performance.

## Key Features
- **Big Convolutional Model**: Over-parameterized architecture to highlight benefits of pruning.
- **Pruning & Downsizing**: L1 unstructured pruning at a global scale.
- **Performance Metrics**: Evaluate speed, memory usage, and accuracy before and after pruning.
- **Inline Documentation**: Clear code structure with inline comments for easy understanding.

## Project Structure
- **src/**
  - `big_model.py`: Defines the large CNN.
  - `train.py`: Training script with Adam optimizer and cross-entropy loss.
  - `prune.py`: Global pruning implementation and sparsity measurement.
  - `utils.py`: Common utilities for data loading or model utilities (if needed).
  - `evaluate.py`: Contains evaluation logic (also used in `train.py`).
- **notebooks/**
  - `big_nn_pruning_experiments.ipynb`: Interactive exploration of training logs, pruning strategies, and performance results.
- **tests/**
  - `test_pruning.py`: Basic test to ensure pruned models can still forward pass and maintain target accuracy.

## Setup & Requirements
```bash
# Create a virtual environment (optional but recommended)
python -m venv mnist-env
source mnist-env/bin/activate  # or mnist-env\Scripts\activate on Windows
```

# Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. **Train the Big Model**  
   ```bash
   python src/train.py
   ```
   After training, a `big_model.pth` file will be saved.

2. **Prune the Model**  
   ```bash
   python src/prune.py
   ```
   A `big_model_pruned.pth` file will be generated, showcasing a much sparser network.

3. **Evaluate Performance**  
   - Evaluate either directly in the training script or via an additional `evaluate.py` script.
   - Check memory footprint and inference speed in `big_nn_pruning_experiments.ipynb`.

## Results
- **Original Model Accuracy**: ~99% on MNIST (depending on hyperparameters).  
- **Pruned Model Accuracy**: ~98-99% with 30-50% of weights removed, significantly reduced memory usage.

## Future Enhancements
- **Quantization**: Combine pruning with post-training quantization for further compression.
- **Structured Pruning**: Evaluate channel-level pruning (removing entire filters).
- **Knowledge Distillation**: Train a smaller "student" model with guidance from the large "teacher" model for even greater compression.