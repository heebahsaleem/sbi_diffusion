import os
import numpy as np
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPSE
import pickle

#to check the timestep, since it is set to 500. Checking where did this come from.
# import sbi
# print("sbi version:", sbi.__version__)


DATA_DIR = "output/simulator"
OUTPUT_DIR = "output/npse"
sinuosity = np.load(
    os.path.join(DATA_DIR, "sinuosity.npy")
)

images = np.load(
    os.path.join(DATA_DIR, "channel_images.npy")
)

print("Sinuosity:", sinuosity.shape)
print("Images:", images.shape)


rng = np.random.default_rng(42)

indices = rng.permutation(len(sinuosity))

n_train = 800

train_idx = indices[:n_train]
test_idx = indices[n_train:]

S_train = sinuosity[train_idx]
X_train = images[train_idx]

S_test = sinuosity[test_idx]
X_test = images[test_idx]

print("Training:", S_train.shape, X_train.shape)
print("Testing :", S_test.shape, X_test.shape)


#since the number of pixels in the images is large, we will compute a summary of the images to reduce the dimensionality of the input data. The summary will be a 1D array that captures the vertical position of the channel in each column of the image.
def compute_summary(image):

    height, width = image.shape
    rows = np.arange(height, dtype=np.float32)

    centroid = np.zeros(width, dtype=np.float32)

    for j in range(width):

        column = image[:, j]

        if column.sum() > 0:
            centroid[j] = (
                rows * column
            ).sum() / column.sum()
        else:
            centroid[j] = height / 2

    return centroid / height

X_train_summary = np.stack(
    [compute_summary(img) for img in X_train]
)

X_test_summary = np.stack(
    [compute_summary(img) for img in X_test]
)

print("Training summaries:", X_train_summary.shape)
print("Test summaries:", X_test_summary.shape)

#convert to tensors 
theta_train = torch.tensor(
    S_train,
    dtype=torch.float32
).unsqueeze(1)

x_train = torch.tensor(
    X_train_summary,
    dtype=torch.float32
)

print("theta:", theta_train.shape)
print("x:", x_train.shape)

#defined prior distribution for the sinuosity parameter. The prior is a uniform distribution over the range [1.7, 1.9].
prior = BoxUniform(
    low=torch.tensor([1.7]),
    high=torch.tensor([1.9])
)

#trainig NPSE model using the training data. The NPSE class is initialized with the prior distribution, and then the training data is appended to the inference object. Finally, the density estimator is trained using the appended simulations.
training = NPSE(prior=prior)

training.append_simulations(
    theta_train,
    x_train
)

density_estimator = training.train()
posterior = training.build_posterior(density_estimator)
print(type(posterior))
print(posterior)

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "posterior.pkl"), "wb") as f:
    pickle.dump(posterior, f)
print("Posterior saved to:", os.path.join(OUTPUT_DIR, "posterior.pkl"))

#save heldout (200) test data to a file for later evaluation of the trained model. The test data is saved as a dictionary containing the test summaries and the corresponding true sinuosity values.
np.save(os.path.join(OUTPUT_DIR, "S_test.npy"), S_test)
np.save(os.path.join(OUTPUT_DIR, "X_test_summary.npy"), X_test_summary)
np.save(os.path.join(OUTPUT_DIR, "X_test_images.npy"), X_test)
print("Saved held-out test data")


#to check the timestep, since it is set to 500. Checking where did this come from.
# import inspect
# print(inspect.signature(posterior.sample))
# print(posterior) #SBI 0.27.0 defaults to 500 sampling steps for VectorFieldPosterior.