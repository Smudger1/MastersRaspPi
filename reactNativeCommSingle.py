import cv2 as cv
import numpy as np
import pickle
from ultralytics import YOLO
from picamzero import Camera

rtsp_URL = "http://192.168.0.169:81/stream"

def setup():


    if not os.path.exists("./objectDetectionImgs"):
        ## Create directory for object detection images if it doesn't exist
        print("Creating directory for object detection images...")
        os.makedirs("./objectDetectionImgs")


    ## Load the YOLO model
    model = YOLO("yolo11n.pt")  # Load the YOLO
    model.export(format="ncnn")  # Export the model to NNC format
    ncnn_model = YOLO("yolo11n_ncnn_model")  # Path to the exported model
    print(f"Model exported to {ncnn_model}")
    print("YOLO model loaded successfully.")

    video1 = Camera()
    video1.still_size = (1280, 720)  # Set the resolution
    video1.video_size = (1280, 720)  # Set the video size

    print("Video stream opened successfully.")

    return ncnn_model, video1

def getRequestedObject():
    # HTTP SERVER REQUEST HANDLING

    # Classes
    ## names:
    ## 0: person
    ## 56: chair
    ## 57: couch
    ## 62: tv
    ## 65: remote
    ## 73: book
    return "tv"  # Example object to track

def main():
    ncnn_model, video1 = setup()
    requested_object = getRequestedObject()
    print(f"Requested object: {requested_object}")
    while True:
        ret, frame1 = video1.read()  # Read frame from the camera

        frame1 = video1.take_photo()

        if not ret:
            print("Error: Could not read frame.")
            break

        results = ncnn_model(frame1)  # Run inference on the first camera frame
        for result in results:
            if result.names[result.class_id] == requested_object:
                print(f"Detected {requested_object} in the first camera.")

        #cv.imshow("Camera Feed", frame1)  # Display the camera feed

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video1.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()