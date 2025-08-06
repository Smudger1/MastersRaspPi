from picamzero import Camera
import os
import cv2 as cv
from time import sleep

rtsp_URL = "http://192.168.0.169:81/stream"

#### Create directory for calibration images if it doesn't exist
if not os.path.exists("./calibImgs"):
    os.makedirs("./calibImgs")

#### Starting the image capture process for both cameras
print("Starting image capture for both cameras...")

master_cam = Camera()
master_cam.still_size = (1280, 720)  # Set the resolution for the camera

slave_cam = cv.VideoCapture(rtsp_URL)  # Adjust the index if necessary

if not slave_cam.isOpened():
    print("Error: Could not open video stream.")
    exit()


for i in range(10):
    sleep(3)  # Wait for 3 seconds between captures
    print(f"Capturing image {i+1} ...")
    master_cam.take_photo(f"./calibImgs/calibImageMaster_{i}.jpg")  # Save the image

    print(f"Captured image {i+1}")

slave_cam.release()
cv.destroyAllWindows()
