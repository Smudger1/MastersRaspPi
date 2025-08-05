import cv2 as cv
import numpy as np
import pickle
from ultralytics import YOLO
from picamzero import Camera
import requests
from time import sleep
from locationCalc import calculateDistance, calculateAngle

rtsp_URL = "http://192.168.0.169:81/stream"

def setup():
    ## Load the YOLO model
    model = YOLO("yolov11n.pt")  # Load the YOLO
    model.export(format="ncnn")  # Export the model to NNC format
    ncnn_model = YOLO("yolo11n_ncnn_model")  # Path to the exported model
    print(f"Model exported to {ncnn_model}")
    print("YOLO model loaded successfully.")

    video1 = Camera()
    video1.still_size = (1280, 720)  # Set the resolution
    video1.video_size = (1280, 720)  # Set the video size

    video2 = cv.VideoCapture(rtsp_URL)  # Open the RTSP stream
    if not video2.isOpened():
        print("Error: Could not open video stream.")
        exit()
    print("Video stream opened successfully.")

    return ncnn_model, video1, video2

def getRequestedObject():
    # HTTP SERVER REQUEST HANDLING

    ## Class names:
    ## 
    ## 0: person
    ## 56: chair
    ## 57: couch
    ## 62: tv
    ## 65: remote
    ## 73: book

    try:
        response = requests.get("http://localhost:5000/requested_object")
        if response.status_code == 200:
            print("Requested object received from server.")
            return response.json().get("requestedObject", None)  # Default to None if not found
        else:
            print("Error: Could not retrieve requested object.")
            return None
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return None
    return None


def main():
    ncnn_model, video1, video2 = setup()
    requested_object = getRequestedObject()

    personResult = None
    objectResult = None

    if not requested_object:
        print("No requested object to track.")
        return None

    while True:
        while requested_object is not None:
            ret, frame1 = video1.read()  # Read frame from the camera
            ret2, frame2 = video2.read()  # Read frame from the RTSP stream

            if not ret and not ret2:
                print("Error: Could not read frames.")
                break

            results = ncnn_model(frame1)  # Run inference on the first camera frame
            for result in results:
                if result.names[result.class_id] == 'person' and personResult is None:
                    personResult = result
                    print("Person detected in the first camera.")
                if result.names[result.class_id] == requested_object and objectResult is None:
                    objectResult = result
                    print(f"Detected {requested_object} in the first camera.")

            if personResult is not None and objectResult is not None:
                print("Found both person and requested object in master camera.")
            else:
                print("Error: Person or requested object not found in master camera.")
                print("Looking in second camera... ")

                personResult = None
                objectResult = None

                results = ncnn_model(frame2)  # Run inference on the second camera frame
                for result in results:
                    if result.names[result.class_id] == 'person' and personResult is None:
                        personResult = result
                        print("Person detected in the second camera.")
                    if result.names[result.class_id] == requested_object and objectResult is None:
                        objectResult = result
                        print(f"Detected {requested_object} in the second camera.")

            if personResult is not None and objectResult is not None:
                print("Calculating angle and distance...")
                person_coords = personResult.boxes.xyxy[0].numpy()
                object_coords = objectResult.boxes.xyxy[0].numpy()
                angle = calculateAngle(person_coords, object_coords)



            cv.imshow("Camera Feed", frame1)  # Display the camera feed
            cv.imshow("RTSP Stream", frame2)  # Display the RTSP stream


            ## Check if requested object has been found
            requested_object = getRequestedObject()

            sleep(0.5) # Wait for a short period before the next iteration

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("Requested object is None, waiting for new request...")
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    


    video1.release()
    video2.release()
    cv.destroyAllWindows()