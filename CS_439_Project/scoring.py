import numpy as np
import matplotlib.pyplot as plt
from data import dataset


def compute_stddev(x_i):
    """"Scoring based on standard deviation"""

    d = x_i.shape[0]
    mu = np.sum(x_i) / d
    stddev = np.sqrt(np.sum((x_i - mu) ** 2) / d)
    return stddev


flattened_X = dataset.x_train.reshape((dataset.x_train.shape[0], -1)) 
stddevs = np.array([compute_stddev(x_i) for x_i in flattened_X])


"""
sorted_indices = np.argsort(stddevs)

X_train_sorted = dataset.x_train[sorted_indices]
Y_train_sorted = dataset.y_train[sorted_indices]

print("Top 10 lowest stddevs:", stddevs[sorted_indices[:10]])
print("Top 10 highest stddevs:", stddevs[sorted_indices[990:]])

#SHOW PICTURE
def plot_images(images, title, num=10):
    plt.figure(figsize=(15, 2))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title, fontsize=16)
    plt.show()

# Visualize 10 images with the lowest stddev
lowest_std_images = X_train_sorted[:10]
plot_images(lowest_std_images, title="100 Images with Lowest Stddev")

# Visualize 10 images with the highest stddev
highest_std_images = X_train_sorted[-10:]
plot_images(highest_std_images, title="100 Images with Highest Stddev")
"""