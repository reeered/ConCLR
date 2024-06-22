# ConCLR: Context-Based Contrastive Learning for Scene Text Recognition

This repository contains the code for reproducing the the paper:

**Context-Based Contrastive Learning for Scene Text Recognition**

## Introduction

This project aims to reproduce the results of the paper "Context-Based Contrastive Learning for Scene Text Recognition". The main contribution of this paper is the introduction of a context-based contrastive learning framework to improve the robustness and accuracy of scene text recognition models.

## Requirements

- Python 3.7 or higher
- PyTorch 1.1.0 or higher
- TensorBoard (optional, for monitoring training progress)
- Other dependencies listed in `requirements.txt`

Install the required packages using:

```sh
pip install -r requirements.txt
```

## Project Structure

The project directory is structured as follows:

```
ConCLR/
├── data/                   # Dataset directory
├── models/                 # Model definitions
├── checkpoints/            # Model checkpoints
├── notebooks/              # Jupyter notebooks
├── configs/                # Configuration file directory
├── runs/                   # Traning logs
├── README.md               # This file
├── requirements.txt        # Required packages
└── *.py                    # Python scripts
```

## Training

To train the model, run the following command:

```sh
python train.py --config configs/config.yaml
```

This will start the training process using the configuration specified in `config.yaml`.

## TensorBoard

You can monitor the training process using TensorBoard. Run the following command to start TensorBoard:

```sh
tensorboard --logdir=runs/{experiment_name}
```

Then, open your web browser and go to `http://localhost:6006`.

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all dependencies are installed correctly.
2. Verify the dataset is placed in the correct directory and is properly formatted.
3. Check the configuration file `config.yaml` for any discrepancies.

## Acknowledgements

This project is based on the ABINet (Autonomous, Bidirectional and Iterative) codebase. Thank the authors for providing their implementation and making it publicly available.

## References

- ConCLR: [Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/20245)
- ABINet: [GitHub Repository](https://github.com/FangShancheng/ABINet)
