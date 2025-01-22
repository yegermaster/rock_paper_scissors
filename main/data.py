import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# Load the dataset
dataset, dataset_info = tfds.load("rock_paper_scissors", as_supervised=True, with_info=True)

# Get train dataset
train_dataset = dataset['train']

# Function to display images
def visualize_dataset(dataset, label_map, num_images=9):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(num_images)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(label_map[label.numpy()])
        plt.axis("off")
    plt.show()

# Label map
label_map = {0: "Rock", 1: "Paper", 2: "Scissors"}

# Visualize the dataset
visualize_dataset(train_dataset, label_map)
