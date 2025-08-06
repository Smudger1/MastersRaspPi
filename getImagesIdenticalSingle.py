from picamzero import Camera
import os
import cv2 as cv
from time import sleep

#### Create directory for calibration images if it doesn't exist
if not os.path.exists("./calibImgs"):
    os.makedirs("./calibImgs")

#### Starting the image capture process for Master cameras
print("Starting image capture for Master cameras...")

master_cam = Camera()
master_cam.still_size = (1280, 720)  # Set the resolution for the camera



for i in range(10):
    sleep(3)  # Wait for 3 seconds between captures
    print(f"Capturing image {i+1} ...")
    master_cam.take_photo(f"./calibImgs/calibImageMaster_{i}.jpg")  # Save the image

    print(f"Captured image {i+1}")

cv.destroyAllWindows()
