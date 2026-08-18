"""
This input parameters and generated 64x256 facies image using this simulaor
"""

import os

import numpy as np
import matplotlib.pyplot as plt


def simulate_channel(
    sinuosity,
    seed=0,
    height=64,
    width=256,
    channel_thickness=8
):
    """
    Simplified 2D fluvial-channel simulator.

    Parameters: 
    
    sinuosity : Controls how strongly the channel meanders.
        first experiment: 1.7 <= sinuosity <= 1.9

    seed :Controls stochastic variation between realizations.

    height, width :output image dimensions.

    channel_thickness : Approximate thickness of the channel in pixels.

    Returns:

    facies :0 for background/shale, 1channel
    """

    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, width)

    amplitude = 7 + 30 * (sinuosity - 1.7) # this is a toy relationship, not an RMS reproduction.

    phase = rng.uniform(0, 2 * np.pi)
    phase2 = rng.uniform(0, 2 * np.pi)

    centerline = (
        height / 2
        + amplitude * np.sin(x + phase)
        + 2.5 * np.sin(2.3 * x + phase2)
    )

    centerline += rng.normal(0, 0.7, size=width)

    facies = np.zeros((height, width), dtype=np.float32)

    half_thickness = channel_thickness // 2

    for j in range(width):
        center = int(np.clip(centerline[j], 0, height - 1))

        top = max(0, center - half_thickness)
        bottom = min(height, center + half_thickness + 1)

        facies[top:bottom, j] = 1.0

    return facies



# #testing several sinuosity values

# values = [1.70, 1.75, 1.80, 1.85, 1.90]

# for i, s in enumerate(values):
#     image = simulate_channel(
#         sinuosity=s,
#         seed=42 + i
#     )

#     plt.figure(figsize=(10, 3))
#     plt.imshow(image, aspect="auto", origin="lower")
#     plt.title(f"Synthetic channel — sinuosity = {s:.2f}")
#     plt.xlabel("Horizontal position")
#     plt.ylabel("Vertical position")
#     plt.tight_layout()
#     plt.savefig(
#         f"synthetic_channel_sinuosity_{s:.2f}.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
#     plt.show()
output_dir = "output/simulator"
os.makedirs(output_dir, exist_ok=True)
N = 1000

sinuosity_values = []
images = []

rng = np.random.default_rng(123)

for i in range(N):

    # Sample sinuosity from the prior
    s = rng.uniform(1.7, 1.9)

    # Different stochastic realization
    seed = i

    image = simulate_channel(
        sinuosity=s,
        seed=seed
    )

    sinuosity_values.append(s)
    images.append(image)

sinuosity_values = np.array(sinuosity_values, dtype=np.float32)
images = np.stack(images).astype(np.float32)

print("Parameters:", sinuosity_values.shape)
print("Images:", images.shape)

np.save(os.path.join(output_dir, "sinuosity.npy"), sinuosity_values)
np.save(os.path.join(output_dir, "channel_images.npy"), images)