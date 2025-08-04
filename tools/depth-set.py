import os
import shutil

# Define the source directory and the destination directory
source_dir = 'path/to/source/directory'
destination_dir = 'path/to/destination/directory'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Loop through the files in the source directory
for file_name in os.listdir(source_dir):
    # Check if the file name contains the keyword "depth"
    if "depth" in file_name.lower() and file_name.endswith('.png'):
        # Construct full file paths
        source_file = os.path.join(source_dir, file_name)
        destination_file = os.path.join(destination_dir, file_name)
        
        # Copy the file to the destination directory
        shutil.copy2(source_file, destination_file)
        print(f"Copied: {file_name}")

print("All matching files have been copied.")