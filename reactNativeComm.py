import cv2 as cv
import numpy as np
import pickle
from ultralytics import YOLO
from picamzero import Camera
import requests
from time import sleep
from locationCalc import calculateDistanceBetweenObjects, calculateAngle, calculateCenter
import os
import sys
sys.path.append('/Depth-Anything-V2')
from getDepthMap import getDepthMap

rtsp_URL = "http://192.168.68.61:81/stream"

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
    """
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
        return None"""
    return "tv"

def sendDistance(distance):
    try:
        response = requests.post("http://localhost:5000/update_distance", json={"distance": distance})
        if response.status_code == 200:
            print("Distance sent successfully.")
        else:
            print(f"Error sending distance: {response.json().get('message', 'Unknown error')}")
    except requests.RequestException as e:
        print(f"HTTP Request failed: {e}")


def main():
    ncnn_model, video1, video2 = setup()
    print("Setup complete.")
    requested_object = getRequestedObject()
    print(f"Requested object: {requested_object}")

    personResult = None
    objectResult = None
    cameraFeed = None

    while True:
        while requested_object is not None:
            print(f"Tracking {requested_object}...")

            # Read frame from both cameras
            photo_path = "./objectDetectionImgs/frame1.jpg"
            video1.take_photo(photo_path)
            frame1 = cv.imread(photo_path)
            ret2, frame2 = video2.read()  # Read frame from the RTSP stream

            if frame1 is not None and not ret2:
                print("Error: Could not read frames.")
                break
            
            results = ncnn_model(frame1)  # Run inference on the first camera frame
            for result in results:
                # result.boxes.cls is a tensor of class indices
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                for class_id in class_ids:
                    class_name = result.names[class_id]
                    if class_name == requested_object and objectResult is None:
                        print(f"Detected {requested_object} in the first camera.")
                        objectResult = result
                    elif class_name == 'person' and personResult is None:
                        personResult = result
                        print("Person detected in the first camera.")


            if personResult is not None and objectResult is not None:
                print("Found both person and requested object in master camera.")
                cameraFeed = 1
            else:
                print("Error: Person or requested object not found in master camera.")
                print("Looking in second camera... ")

                personResult = None
                objectResult = None

                results = ncnn_model(frame2)  # Run inference on the second camera frame
                for result in results:
                    # result.boxes.cls is a tensor of class indices
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    for class_id in class_ids:
                        class_name = result.names[class_id]
                        if class_name == requested_object and objectResult is None:
                            print(f"Detected {requested_object} in the second camera.")
                            objectResult = result
                        elif class_name == 'person' and personResult is None:
                            personResult = result
                            print("Person detected in the second camera.")
                
                if personResult is not None and objectResult is not None:
                    print("Found both person and requested object in slave camera.")
                    cameraFeed = 2

            if personResult is not None and objectResult is not None:
                print("Calculating angle and distance...")
                person_coords = calculateCenter(personResult.boxes.xyxy[0].numpy())
                object_coords = calculateCenter(objectResult.boxes.xyxy[0].numpy())
                angle = calculateAngle(person_coords, object_coords)

                print(f"Angle between person and {requested_object}: {angle} degrees")

                if cameraFeed == 1:
                    depthMap = getDepthMap(frame1)
                else:
                    depthMap = getDepthMap(frame2)
                distanceToPerson = depthMap[int(person_coords[1]), int(person_coords[0])]
                distanceToObject = depthMap[int(object_coords[1]), int(object_coords[0])]

                distance = calculateDistanceBetweenObjects(angle, distanceToPerson, distanceToObject)
                print(f"Distance between person and {requested_object}: {distance} meters")
                ##sendDistance(distance)
            else:
                print("ERROR: Person or requested object not found in either camera.")
                break

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

if __name__ == "__main__":
    main()
