import numpy as np
import matplotlib.pyplot as plt

S = np.load("/Home/siv36/hesal5042/Research/NORCE/sbi_diffusion/output/simulator/sinuosity.npy")
X = np.load("/Home/siv36/hesal5042/Research/NORCE/sbi_diffusion/output/simulator/channel_images.npy")

print("S shape:", S.shape)
print("X shape:", X.shape)

print("Sinuosity min:", S.min())
print("Sinuosity max:", S.max())
print("Image values:", np.unique(X))

# Show 6 random simulations
rng = np.random.default_rng(42)
indices = rng.choice(len(S), 6, replace=False)

for idx in indices:
    plt.figure(figsize=(10, 3))
    plt.imshow(X[idx], origin="lower", aspect="auto")
    plt.title(f"Sinuosity = {S[idx]:.3f}")
    plt.xlabel("Horizontal position")
    plt.ylabel("Vertical position")
    plt.tight_layout()
    plt.show()