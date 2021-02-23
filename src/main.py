import os
import zipfile
import numpy as np
import tensorflow as tf

from manipulators.preprocessor import *
from manipulators.data_augmenter import *

from tensorflow import keras
from tensorflow.keras import layers

# Make a directory to store the data.
if not os.path.exists("data"):
	os.makedirs("data")

# Download url of normal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "./data/CT-0.zip")
keras.utils.get_file(filename, url)

# Download url of abnormal CT scans.
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename = os.path.join(os.getcwd(), "./data/CT-23.zip")
keras.utils.get_file(filename, url)

# Unzip the scans into the data directory
with zipfile.ZipFile("./data/CT-0.zip", "r") as z_fp:
	z_fp.extractall("./data/")

with zipfile.ZipFile("./data/CT-23.zip", "r") as z_fp:
	z_fp.extractall("./data/")

# Folder "CT-0" consist of CT scans having normal lung tissue,
# no CT-signs of viral pneumonia.
normal_scan_paths = [
	os.path.join(os.getcwd(), "data/CT-0", x)
	for x in os.listdir("data/CT-0")
]
# Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# involvement of lung parenchyma.
abnormal_scan_paths = [
	os.path.join(os.getcwd(), "data/CT-23", x)
	for x in os.listdir("data/CT-23")
]

print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))

# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
normal_scans = np.array([process_scan(path) for path in normal_scan_paths])

print("Done")
# For the CT scans having presence of viral pneumonia
# assign 1, for the normal ones assign 0.
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Split data in the ratio 70-30 for training and validation.
x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
print(
	"Number of samples in train and validation are %d and %d."
	% (x_train.shape[0], x_val.shape[0])
)

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
	train_loader.shuffle(len(x_train))
	.map(train_preprocessing)
	.batch(batch_size)
	.prefetch(2)
)
# Only rescale.
validation_dataset = (
	validation_loader.shuffle(len(x_val))
	.map(validation_preprocessing)
	.batch(batch_size)
	.prefetch(2)
)

