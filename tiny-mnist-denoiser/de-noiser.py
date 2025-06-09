"""
Copyright 2025 Victor Steinborn

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, random
from jax.nn import sigmoid
from jax.tree_util import tree_map
from torch.utils.data import DataLoader, default_collate, Dataset
from torchvision.datasets import MNIST
import time
import matplotlib.pyplot as plt
import pickle

# Hyper Parameters
layer_sizes = [784, 32, 784]
step_size = 0.001
num_epochs = 1000
batch_size = 256

# Initialize neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize neural network parameters with correct sizes
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# Activation Function
def relu(x):
    return jnp.maximum(0, x)

# Network Definition
def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = sigmoid(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return sigmoid(logits)

# Initialize parameters
params = init_network_params(layer_sizes, random.key(0))

# Test on a single example
def test_single_image():
    random_flattened_image = random.normal(random.key(1), (28 * 28,))
    preds = predict(params, random_flattened_image)
    assert preds.size == 28*28, "Prediction output format incorrect"
test_single_image()

# Handle batches
batched_predict = vmap(predict, in_axes=(None, 0))

# Test on a batch of images
def test_batched_images():
    random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
    batched_preds = batched_predict(params, random_flattened_images)
    assert batched_preds.shape == (10, 28*28), "Batched prediction output format incorrect"
test_batched_images()

def mean_square_error(params, images, targets):
    predictions = batched_predict(params, images)
    delta = jnp.linalg.norm(predictions - targets, axis = 1)
    return jnp.mean(delta)

def loss(params, images, targets):
    predictions = batched_predict(params, images)
    cross_entropy_sum = jnp.sum(targets*jnp.log(predictions) + (1-targets)*jnp.log(1-predictions), axis=1)
    return -jnp.mean(cross_entropy_sum)

# arXiv:1412.6980v9
def adam_step(t, grads, state, b1=0.9, b2=0.999, eps=1e-8):
    params, m, v = state
    m = b1*m + (1-b1)*grads
    v = b2*v + (1-b2)*jnp.square(grads)
    mhat = m/(1-b1**t)
    vhat = v/(1-b2**t)
    params = params - step_size*mhat/(jnp.sqrt(vhat)+eps)
    return params, m, v

@jit
def update(t, state, x, y):
    params, ms, vs = state
    grads = grad(loss)(params, x, y)
    updated_params = []
    updated_ms=[]
    updated_vs=[]
    for (w, b), (grad_w, grad_b), (mw, mb), (vw, vb) in zip(params, grads, ms, vs):
        state_w = (w, mw, vw)
        state_b = (b, mb, vb)
        new_state_w = adam_step(t, grad_w, state_w)
        new_state_b = adam_step(t, grad_b, state_b)
        updated_params.append((new_state_w[0], new_state_b[0]))
        updated_ms.append((new_state_w[1], new_state_b[1]))
        updated_vs.append((new_state_w[2], new_state_b[2]))
    updated_state = updated_params, updated_ms, updated_vs
    return updated_state

def numpy_collate(batch):
    # Collate function specifies how to combine a list of data samples into a batch.
    # default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    return tree_map(np.asarray, default_collate(batch))

def flatten_and_cast(pic):
    # Convert PIL image to flat (1-dimensional) numpy array.
    return np.ravel(np.array(pic, dtype=jnp.float32))

class AutoMNISTDataset(Dataset):
    def __init__(self, data, lables):
        self.data = data
        self.lables = lables

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = (self.data[index], self.lables[index])
        return item

def image_preprocessing(dataset, duplicated_origionals=1):
    flattend_and_normed = np.array(dataset.data.numpy().reshape(len(dataset.data), -1))/255
    return np.repeat(flattend_and_normed, duplicated_origionals, axis=0)

def image_noising(dataset):
    noise = np.random.default_rng(seed=1).uniform(low=-1,high=1, size=dataset.shape)
    return dataset + noise

# Get the train dataset for evaluation
mnist_dataset = MNIST('./data/mnist/', download=True, transform=flatten_and_cast)
# training_generator = DataLoader(mnist_dataset, batch_size=batch_size, collate_fn=numpy_collate)
train_labels = image_preprocessing(mnist_dataset)
train_images = image_noising(train_labels)

mnist_auto_dataset = AutoMNISTDataset(train_images, train_labels)
training_generator = DataLoader(mnist_auto_dataset, batch_size=batch_size, collate_fn=numpy_collate)

# Get the test dataset for evaluation
mnist_dataset_test = MNIST('./data/mnist/', download=True, train=False)
test_labels = jnp.array(image_preprocessing(mnist_dataset_test))
test_images_clean = jnp.array(test_labels)
test_images = jnp.array(image_noising(test_labels))

# Train the model
total_start_time = time.time()
t=1
state = params, [(jnp.zeros_like(w0), jnp.zeros_like(b0)) for w0, b0 in params], [(jnp.zeros_like(w0), jnp.zeros_like(b0)) for w0, b0 in params]
for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in training_generator:
        state = update(t, state, x, y)
        t+=1
    epoch_time = time.time() - start_time

    train_msq = mean_square_error(state[0], train_images, train_labels)
    test_msq = mean_square_error(state[0], test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set mean square error {}".format(train_msq))
    print("Test set mean square error {}".format(test_msq))
training_time = time.time() - total_start_time
print("Total training time: {:}".format(training_time))

# Save parameters
with open('./out/de-noiser/parameters.pkl', 'wb') as f:
    pickle.dump(state[0], f)

# See result

def unflatten(flat_array, shape):
    return np.resize(flat_array, shape)

def visualize_images(images, predictions, originals):
    """
    Function to display multiple image pairs from numpy arrays side by side.
    """
    rows = 3
    cols = 10

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2))

    for i in range(cols):
        plt.imsave(f"./out/de-noiser/{i}-noise.png",unflatten(images[i], (28, 28)), cmap='grey')
        axs[0,i].imshow(unflatten(images[i], (28, 28)), cmap='grey')
        plt.imsave(f"./out/de-noiser/{i}-prediction.png",unflatten(predictions[i], (28, 28)), cmap='grey')
        axs[1,i].imshow(unflatten(predictions[i], (28, 28)), cmap='grey')
        plt.imsave(f"./out/de-noiser/{i}-original.png",unflatten(originals[i], (28, 28)), cmap='grey')
        axs[2,i].imshow(unflatten(originals[i], (28, 28)), cmap='grey')

    fig.suptitle("Image comparisons")
    plt.show()

random_indices = np.random.choice(len(test_images), size=10, replace=False)
visualize_images([test_images[i] for i in random_indices], [predict(state[0], test_images[i]) for i in random_indices], [test_images_clean[i] for i in random_indices])
