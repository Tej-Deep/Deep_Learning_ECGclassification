import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from random import randint, choice
from preprocess.datasets import AugmentedDataset
import torch
import sys
sys.path.append('../')


def generate_indices(p, pop_size):
    dist = torch.distributions.Bernoulli(p)

    return dist.sample((pop_size,)).nonzero().squeeze()


def ecg_random_mask(signal, probablities=[0.3, 0.15], mask_width=[0.08, 0.18], mask_value=0.0, sampling_rate=100):
    """Random masking of ECG signals

    Args:
        signal (_type_): _description_
        probablities (list, optional): Probabilities of masking signals, first value for batch dimension, second value for lead dimension. Defaults to [0.3, 0.15].
        mask_width (list, optional): Width range of masking windows (in seconds). Defaults to [0.08, 0.18].
        mask_value (float, optional): Value to mask signal with. Defaults to 0.0.
        sampling_rate (int, optional): Sampling rate of signal. Defaults to 100.
    """
    batch, lead, length = signal.shape
    mask_width = (np.array(mask_width) * sampling_rate).round().astype(int)

    sig = signal.clone()

    sig_mask_prob = probablities[1] / mask_width[1]

    mask = torch.full_like(sig, 1, dtype=sig.dtype, device=sig.device)

    # randomly select batch indices to mask
    batch_indices = generate_indices(probablities[0], batch)
    masks = []
    for b_idx in batch_indices:
        indices = np.array(generate_indices(
            sig_mask_prob, length-mask_width[1]), ndmin=1)
        mask_list = []
        indices += mask_width[1]//2

        for j in indices:
            masked_radius = randint(mask_width[0], mask_width[1])//2
            mask[b_idx, :, j-masked_radius:j+masked_radius] = mask_value
            mask_list.append((j-masked_radius, j+masked_radius))

        masks.append(mask_list)

    sig = sig.mul_(mask)

    return sig, batch_indices, masks


def visualize_masking(modified_batches, masks, inputs, outputs):
    """Visualize a random lead of a random batch from the modified batches using matplotlib

    Args:
        modified_batches (list): List of indices modified from the batch
        masks (list[list[tuple]]): List of list of ranges of masked indices
        inputs (list[list[list]]): Inputs given in batch x sample x lead
        outputs (list[list[list]]): Inputs given in batch x sample x lead
        idx (int, optional): Index of the batch to visualize. Defaults to -1 (random).
    """
    i = choice(range(len(modified_batches)))
    idx = modified_batches[i]
    sample = inputs[idx:idx+1, :, 1]
    sample_masks = masks[i]
    sq = sample.flatten()
    plt.plot(sq)
    plt.title("Original ECG")
    plt.show()

    masked_sample = outputs[idx:idx+1, :, 1]
    msq = masked_sample.flatten()
    for r in sample_masks:
        rect = patches.Rectangle(
            (r[0], min(msq)), r[1]-r[0], max(msq)-min(msq), alpha=0.2, facecolor='red')
        plt.gca().add_patch(rect)
    plt.plot(msq)
    plt.title("Masked ECG")
    plt.show()


def generate_samples_rm(train_loader, visualize=False, min_samples=0, max_samples=None):
    samples_generated = 0
    generated_X = []
    generated_Y = []
    generated_notes = []

    while samples_generated <= min_samples:
        for batch in train_loader:
            inputs, outputs, text = batch
            # inputs are given in batch, sample, lead
            masked, modified_batches, masks = ecg_random_mask(
                inputs.permute(0, 2, 1))
            masked = masked.permute(0, 2, 1)
            samples_generated += len(modified_batches)
            if visualize:
                visualize_masking(modified_batches, masks, inputs, outputs)

            # Add augmented samples
            for i in modified_batches:
                generated_X.append(masked[i])
                generated_Y.append(outputs[i])
                generated_notes.append(text[i])

            if max_samples is not None and samples_generated > max_samples:
                break

    if max_samples is not None:
        generated_X = generated_X[:max_samples]
        generated_Y = generated_Y[:max_samples]
        generated_notes = generated_notes[:max_samples]

    dl = AugmentedDataset(torch.stack(generated_X), torch.stack(generated_Y), torch.stack(generated_notes))

    return dl
