# Vanilla GAN — MNIST

A from-scratch implementation of Vanilla GAN trained on MNIST using PyTorch.

## Results

**Mode Collapse observed at ~Step 6 (epoch ~300)**

Early epochs showed variety (1s, 7s, 9s). By epoch 300+, 
G collapsed to generating only "1" across all 32 fixed noise outputs.

D_loss dropped to ~0.09 while G_loss remained high (~2.9), 
confirming D dominated and G stopped exploring.

[insert GIF or side-by-side image of early vs collapsed output]

## What I learned

- Vanilla GAN has no mechanism to prevent mode collapse
- Very low D_loss is a bad sign — means G is too predictable
- Ideal GAN training: D_loss ~0.5-0.7, G_loss ~1.0-2.0
- No BatchNorm + fully connected only = unstable training

## Why this motivates DCGAN

DCGAN fixes this with BatchNorm, strided convolutions, 
and a more balanced architecture → see DCGAN notebook

## Architecture
[your architecture details]

## Training
[hyperparams]
