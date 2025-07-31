from picamzero import Camera
import os
import cv2 as cv

rtsp_URL = "http://192.168.0.169:81/stream"

#### Create directory for calibration images if it doesn't exist
if not os.path.exists("./calibImgs"):
    os.makedirs("./calibImgs")

#### Capture calibration images for Master camera (Raspberry Pi Camera)

cam = Camera()

cam.start_preview()
cam.capture_sequence(f"./calibImgs/calibImageMaster.jpg", num_images=10, interval=2) 
cam.stop_preview()


#### Capture calibration images for Slave camera (ESP32-CAM)

cap = cv.VideoCapture(rtsp_URL)  # Adjust the index if necessary

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    cv.imwrite(f"./calibImgs/calibImageSlave-{i}.jpg", frame)
    cv.waitKey(2000)  # Wait for 2 seconds between captures

cap.release()
cv.destroyAllWindows()