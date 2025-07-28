from picamzero import Camera
import os


cam = Camera()

cam.start_preview()
cam.take_photo(f"./testImages/new_image.jpg") #save the image to your desktop
cam.stop_preview()