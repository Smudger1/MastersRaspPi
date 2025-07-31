from picamzero import Camera
import os
import cv2 as cv


cam = Camera()
cam.still_size = (1280, 720)  # Set the resolution for the camera

##cam.start_preview()
cam.take_photo(f"setupImg.jpg") #save the image to your desktop
##cam.stop_preview()

slave_cam = cv.VideoCapture("http://192.168.0.169:81/stream")
ret, frame = slave_cam.read()
if not ret:
    print("Error: Could not read frame.")
    
else:
    cv.imwrite(f"calibImageSlaveTEST.jpg", frame)

slave_cam.release()
cv.destroyAllWindows()