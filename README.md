# Bachelor Thesis

## Overview

This project was undertaken during the final year of my degree under the supervision of Prof. Giampaolo Liuzzi.

Bayesian Optimization involves optimizing a function by selecting input values from a feasible set without using derivatives. It's ideal for black-box functions requiring significant evaluation time. This project uses Ax, a wrapper for BOTorch, to efficiently perform Bayesian Optimization with GPU acceleration. The algorithm optimizes test functions and tunes CNN hyperparameters for CIFAR-10 image classification using PyTorch.

## Directory Structure

- `ax_cifar10_restart.py`: Script to restart training with optimized hyperparameters.
- `cifar10.json`: JSON file containing configuration settings for Ax.
- `cifar10.py`: Main script for loading CIFAR-10 data and training the model.
- `cnet.py`: Defines the convolutional neural network architecture.
- `find_in_csv.py`: Script to find specific entries in a CSV file.
- `LICENSE`: License information for this project.
- `README.md`: Project documentation (this file).
- `requirements.txt`: List of Python dependencies required for the project.
- `ReturnThread.py`: Utility for handling return values from threads.
- `stats.csv`: CSV file containing statistics from model training.

## Getting Started

### Prerequisites

Make sure you have Python installed. It's recommended to create a virtual environment for the project to manage dependencies.

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/lapry34/bachelor_thesis.git
   cd bachelor_thesis
2. **Install PyTorch**:
    Visit the [PyTorch website](https://pytorch.org/get-started/locally/) and follow the instructions to install the version compatible with your system.

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
### Running the Project

1. **Train the model**:
   ```sh
   python ax_cifar10_restart.py
   ```

2. **Print best hyperparameters**:
   ```sh
   python find_in_csv.py
   ```

### Scripts Description

- **cifar10.py**: Loads the CIFAR-10 dataset, defines the neural network, and trains the model.
- **ax_cifar10_restart.py**: Uses Ax to optimize hyperparameters and restart the training process.
- **cnet.py**: Contains the definition of the convolutional neural network used for CIFAR-10 classification.
- **find_in_csv.py**: Utility script to search for specific entries within `stats.csv`.
- **ReturnThread.py**: A utility for running threads that return values, facilitating concurrent execution.

## Project Details

### Model

The project uses a Convolutional Neural Network (CNN) for image classification. The architecture and training process are defined in `cnet.py` and `cifar10.py`.

### Hyperparameter Optimization

Hyperparameters are optimized using Ax, a platform for optimizing machine learning models developed by Meta. The optimization process is handled in `ax_cifar10_restart.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to [Prof. Liuzzi](https://sites.google.com/diag.uniroma1.it/giampaolo-liuzzi/home) for the guidance and support throughout the project.
