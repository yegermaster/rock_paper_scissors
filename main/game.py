import tensorflow as tf
import numpy as np
import random


# Label map
label_map = {0: "Rock", 1: "Paper", 2: "Scissors"}


model = tf.keras.models.load_model("rps_model.h5")
# Function to classify an image
def classify_image(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return label_map[np.argmax(predictions)]

# Simple game loop
def play_game(model):
    user_image_path = input("Enter the path to your image (Rock/Paper/Scissors): ")
    user_choice = classify_image(user_image_path, model)
    print("You played:", user_choice)
    computer_choice = random.choice(["Rock", "Paper", "Scissors"])
    print("Computer played:", computer_choice)

    if user_choice == computer_choice:
        print("It's a draw!")
    elif (user_choice == "Rock" and computer_choice == "Scissors") \
         or (user_choice == "Paper" and computer_choice == "Rock") \
         or (user_choice == "Scissors" and computer_choice == "Paper"):
        print("You win!")
    else:
        print("Computer wins!")

# Example usage after training
for i in range(10):
    play_game(model)
