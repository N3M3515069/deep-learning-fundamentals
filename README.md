# dl-progression

Progressive PyTorch implementations built from scratch — every layer, loss function, and training loop written manually without high-level abstractions.

## Notebooks
| Notebook | Description |
|----------|-------------|
| Vanilla_NN_MNIST | Feedforward neural network on MNIST, 95% test accuracy |
| CNN_FashionMNIST | CNN on FashionMNIST, 88.7% test accuracy |
| Vanilla_GAN_MNIST | Vanilla GAN on MNIST — mode collapse observed at epoch ~300 |
| DCGAN_MNIST | DCGAN on MNIST — convolutional GAN, stable training, diverse digit generation |
| SRGAN_DIV2K | Super Resolution GAN — upscaling low-res images using perceptual loss | 🔄 Coming soon |
| ESRGAN_DIV2K | Enhanced SRGAN — RRDB blocks, relativistic discriminator, HuggingFace deployment | 🔄 Coming soon |

## Stack
Python, PyTorch, scikit-learn, matplotlib, torchmetrics
