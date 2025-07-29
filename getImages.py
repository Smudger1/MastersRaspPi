from picamzero import Camera
import os


cam = Camera()

cam.start_preview()
cam.capture_sequence(f"./calibImgs/calibImage.jpg", num_images=10, interval=2) #save the image to your desktop
cam.stop_preview()