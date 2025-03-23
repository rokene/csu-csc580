import os
import face_recognition
from PIL import Image, ImageDraw

# Define input and output directories
input_dir = "input_images"
output_dir = "output_images"

# Clean the output directory
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(output_dir)

# Process each JPG image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        print(f"Processing {image_path}...")

        # Load image
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        print(f"Found {len(face_locations)} face(s) in {filename}")

        # Convert to PIL image
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # Draw red rectangles
        for top, right, bottom, left in face_locations:
            print(f"  Face at Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

        # Save and show the result
        output_path = os.path.join(output_dir, f"faces_{filename}")
        pil_image.save(output_path)
        print(f"Saved annotated image to {output_path}\n")

        # Show the image with faces outlined
        pil_image.show()
