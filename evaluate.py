import os
import pickle
import numpy as np
import torch


#load trained NPSE posterior and all held out test data. The posterior is loaded from a pickle file, and the test data is loaded from numpy files. The test data includes the true sinuosity values, the computed summaries of the test images, and the original test images themselves.
OUTPUT_DIR = "output/npse"
with open(os.path.join(OUTPUT_DIR, "posterior.pkl"), "rb") as f:
    posterior = pickle.load(f)
S_test = np.load(os.path.join(OUTPUT_DIR, "S_test.npy"))
X_test_summary = np.load(os.path.join(OUTPUT_DIR, "X_test_summary.npy"))
print("Number of test channels:", len(S_test))


#run all 200 held out test channels through the trained NPSE model to generate posterior samples for each channel. The posterior samples are generated using the `posterior.sample` method, which takes the number of samples to generate and the observed summary statistics as input. The mean and standard deviation of the posterior samples are computed for each channel and printed to the console.
NUM_SAMPLES = 200

posterior_means = []
posterior_stds = []
all_samples = []

for i in range(len(S_test)):

    # Channel information for this held-out example
    x_obs = torch.tensor(
        X_test_summary[i],
        dtype=torch.float32
    ).unsqueeze(0)

    # Generate possible sinuosity values
    samples = posterior.sample(
        (NUM_SAMPLES,),
        x=x_obs,
        show_progress_bars=False
    )

    samples = samples.squeeze().numpy()
    all_samples.append(samples)

    posterior_means.append(samples.mean())
    posterior_stds.append(samples.std())

    print(
        f"{i+1}/{len(S_test)} | "
        f"True: {S_test[i]:.3f} | "
        f"Mean: {samples.mean():.3f} | "
        f"Std: {samples.std():.3f}"
    )

all_samples = np.array(all_samples)
print("Posterior samples shape:", all_samples.shape)
np.save(os.path.join(OUTPUT_DIR, "eval_posterior_samples.npy"),all_samples)
posterior_means = np.array(posterior_means)
posterior_stds = np.array(posterior_stds)


np.save(os.path.join(OUTPUT_DIR, "posterior_means.npy"), posterior_means)
np.save(os.path.join(OUTPUT_DIR, "posterior_stds.npy"), posterior_stds)

