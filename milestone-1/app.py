import os
import face_recognition
from PIL import Image, ImageDraw

# Define input and output directories
group_dir = "group"
individual_dir = "individual"
output_dir = "output_matches"

# Clean the output directory
if os.path.exists(output_dir):
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(output_dir)

# Load individual faces
individual_encodings = []
individual_images = []
individual_names = []

print("Loading individual images...")
for filename in os.listdir(individual_dir):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(individual_dir, filename)
        image = face_recognition.load_image_file(path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) != 1:
            print(f"Warning: {filename} has {len(face_locations)} faces; skipping.")
            continue

        encoding = face_recognition.face_encodings(image, face_locations)[0]
        individual_encodings.append(encoding)
        individual_images.append(Image.fromarray(image))
        name_without_extension = os.path.splitext(filename)[0]
        individual_names.append(name_without_extension)

print(f"Loaded {len(individual_encodings)} individual(s).\n")

# Track match counts for each individual
individual_match_counts = {name: 0 for name in individual_names}

# Process each group image
for group_filename in os.listdir(group_dir):
    if group_filename.lower().endswith(".jpg"):
        group_path = os.path.join(group_dir, group_filename)
        print(f"Processing group image {group_path}...")

        # Load group image
        group_image_np = face_recognition.load_image_file(group_path)
        group_face_locations = face_recognition.face_locations(group_image_np)
        group_face_encodings = face_recognition.face_encodings(group_image_np, group_face_locations)

        group_image_pil = Image.fromarray(group_image_np)
        group_draw = ImageDraw.Draw(group_image_pil)

        for i, (top, right, bottom, left) in enumerate(group_face_locations):
            group_face_encoding = group_face_encodings[i]

            # Compare against all individuals
            matches = face_recognition.compare_faces(individual_encodings, group_face_encoding, tolerance=0.6)

            for j, match in enumerate(matches):
                if match:
                    individual_name = individual_names[j]
                    individual_match_counts[individual_name] += 1

                    print(f"Match found: {individual_name} in {group_filename}")

                    # Draw rectangle around the matched face
                    group_draw.rectangle([left, top, right, bottom], outline="red", width=3)

                    # Create a combined image
                    individual_image = individual_images[j]

                    # Resize images to the same height
                    max_height = max(individual_image.height, group_image_pil.height)
                    individual_resized = individual_image.resize(
                        (int(individual_image.width * max_height / individual_image.height), max_height)
                    )
                    group_resized = group_image_pil.resize(
                        (int(group_image_pil.width * max_height / group_image_pil.height), max_height)
                    )

                    # Create side-by-side combined image
                    total_width = individual_resized.width + group_resized.width
                    combined_image = Image.new('RGB', (total_width, max_height))
                    combined_image.paste(individual_resized, (0, 0))
                    combined_image.paste(group_resized, (individual_resized.width, 0))

                    # Save the combined image
                    output_filename = f"match_{individual_name}_in_{os.path.splitext(group_filename)[0]}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    combined_image.save(output_path)
                    print(f"Saved match image to {output_path}\n")

# Final match summary
print("\nMatch Summary:")
for name, count in individual_match_counts.items():
    print(f"  {name}: {count} match(es)")
