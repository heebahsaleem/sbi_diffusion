import os
import pickle
import numpy as np
import torch

OUTPUT_DIR = "output/npse"

#load trained NPSE posterior
with open(os.path.join(OUTPUT_DIR, "posterior.pkl"), "rb") as f:
    posterior = pickle.load(f)

#load held out test data
S_test = np.load(os.path.join(OUTPUT_DIR, "S_test.npy"))
X_test_summary = np.load(
    os.path.join(OUTPUT_DIR, "X_test_summary.npy")
)
X_test_images = np.load(
    os.path.join(OUTPUT_DIR, "X_test_images.npy")
)

print("Test sinuosity:", S_test.shape)
print("Test summaries:", X_test_summary.shape)
print("Test images:", X_test_images.shape)


#pick one held out channel
test_idx = 0
true_s = S_test[test_idx]
x_obs = X_test_summary[test_idx]
print("True sinuosity:", true_s)

#converting channel to tensors
x_obs = torch.tensor(
    x_obs,
    dtype=torch.float32
).unsqueeze(0)
print("Observation shape:", x_obs.shape)

#generate samples
samples = posterior.sample((1000,), x_obs)
samples = samples.squeeze().numpy()
print("Samples shape:", samples.shape)

print("\nTrue sinuosity :", true_s)
print("Posterior mean :", samples.mean())
print("Posterior std  :", samples.std())
print("Posterior min  :", samples.min())
print("Posterior max  :", samples.max())