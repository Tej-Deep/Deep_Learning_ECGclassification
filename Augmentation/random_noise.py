import numpy as np
import torch

import random
import matplotlib.pyplot as plt
from preprocess.datasets import AugmentedDataset


def ecg_noising(signal, snr):
    """Noise ECG sample by adding random Gaussian noise. 
    The amount of noise added should be chosen based on the desired signal-to-noise ratio (SNR).

    Args:
        signal (torch.Tensor): Batch of ECGs to be augmented, shape = (batch, lead, length)
        snr (float): Signal-to-noise ratio
    """
    batch, lead, length = signal.shape
    std = torch.std(signal, dim=2, keepdim=True)
    noise_std = snr * std
    noise = np.random.normal(0.0, noise_std, size=(batch, lead, length))
    signal += noise

    return signal


def visualize_noising(batch_size, inputs, noised, zoom=-1):
    """Visualize a random lead of a random batch from the modified batches using matplotlib

    Args:
        batch_size (int): Batch size 
        inputs (torch.Tensor): Original ECGs 
        noised (torch.Tensor): Noised ECGs 
        zoom (int, optional): Zoomed in view. Defaults to -1.
        idx (int, optional): Index of the sample to visualize. Defaults to -1.
    """
    idx = random.randint(0, batch_size)
    sample = inputs[idx:idx+1, :, 1]
    sq = sample.flatten()
    plt.plot(sq)
    plt.title("Original ECG")
    plt.show()

    noised_sample = noised[idx:idx+1, :, 1]
    ns = noised_sample.flatten()
    plt.plot(ns)
    plt.title("Noised ECG")
    plt.show()

    if zoom > 0:

        sample = inputs[idx:idx+1, :zoom, 1]
        sq = sample.flatten()
        plt.plot(sq)
        plt.title("Original ECG (zoomed in)")
        plt.show()

        noised_sample = noised[idx:idx+1, :zoom, 1]
        ns = noised_sample.flatten()
        plt.plot(ns)
        plt.title("Noised ECG (zoomed in)")
        plt.show()


def generate_samples_noising(train_loader, batch_size, visualize=False, zoom=-1, min_samples=0, max_samples=None, snr=0.01):
    """ Generate samples by adding random Gaussian noise to the original samples

    Args:
        train_loader (Dataloader): Dataloader of the original dataset
        batch_size (int): Batch size
        visualize (bool, optional): To plot ECGs. Defaults to False.
        zoom (int, optional): Zoomed in view of ECGs. Defaults to -1.
        min_samples (int, optional): Minimum number of samples to generate. Defaults to 0.
        max_samples (_type_, optional): Maximum number of samples to return. Defaults to None.
        snr (float, optional): Signal to noise ratio. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    samples_generated = 0

    generated_X = []
    generated_Y = []
    generated_notes = []

    while samples_generated <= min_samples:
        for batch in train_loader:
            inputs, outputs, text = batch
            noised = ecg_noising(inputs.permute(0, 2, 1), snr=snr)
            noised = noised.permute(0, 2, 1)

            samples_generated += len(outputs)

            if visualize:
                visualize_noising(batch_size, inputs,
                                  noised, zoom)

            generated_X.extend(noised)
            generated_Y.extend(outputs)
            generated_notes.extend(text)

            if max_samples is not None and samples_generated > max_samples:
                break

    if max_samples is not None:
        generated_X = generated_X[:max_samples]
        generated_Y = generated_Y[:max_samples]
        generated_notes = generated_notes[:max_samples]

    dl = AugmentedDataset(generated_X, generated_Y, generated_notes)

    return dl
