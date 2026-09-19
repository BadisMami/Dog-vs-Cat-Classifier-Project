# Dog vs. Cat Classifier

A convolutional neural network trained from scratch (no pretrained weights) to tell cats from dogs, built in PyTorch.

## How it works

- `Model.ipynb` loads the `archive/dogcat` dataset with `torchvision.datasets.ImageFolder`, resizes images to 128x128, and augments the training split with random flips, rotation, and color jitter.
- The network is a small CNN: two conv + max-pool blocks feeding into two fully connected layers, ending in a single sigmoid output for cat (0) vs. dog (1).
- It's trained for 60 epochs with SGD and binary cross-entropy loss.
- On the held-out test split, it reaches about 84% accuracy.

## Running it

Open `Model.ipynb` in Jupyter with PyTorch, torchvision, and matplotlib installed, and run the cells in order. The dataset paths in the notebook are absolute, update them to point at your local copy of `archive/dogcat`.
