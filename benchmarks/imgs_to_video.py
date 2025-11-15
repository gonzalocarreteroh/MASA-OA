import cv2
import os
from glob import glob

# Folder where your images are located
image_folder = '.\MOT15\Venice-2\img1'

# Get all .jpg images and sort them by name
images = sorted(glob(os.path.join(image_folder, '*.jpg')))

# Read the first image to get dimensions
frame = cv2.imread(images[0])
height, width, layers = frame.shape

# Output video file
output_path = '.\MOT15\Venice-2\Venice-2.mp4'
fps = 30

# Define the video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' or 'XVID'
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write each frame
for image in images:
    frame = cv2.imread(image)
    video.write(frame)

# Release everything
video.release()

print("Video saved to", output_path)