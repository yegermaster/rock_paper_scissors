from rembg import remove
from PIL import Image

input_path = "test_images/s4.jpeg"
output_path = "test_images/s4_rembg.png"

print("Opening the input image...")
input = Image.open(input_path)

print("Removing the background. This might take a few moments...")
output = remove(input, bgcolor=(255, 255, 255, 255))

print("Saving the output image...")
output.save(output_path)

print(f"Process complete! The output has been saved to '{output_path}'.")
