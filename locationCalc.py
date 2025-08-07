import cv2 as cv
import numpy as np
import pickle

def calculateDistance(obj1_pixel_coords, obj2_pixel_coords, obj1_distance, obj2_distance):
    """
    Calculate the distance between two objects in an image based on their pixel coordinates and known distances.
    
    :param imageLoc: Path to the image file.
    :param obj1_pixel_coords: Pixel coordinates of the first object (x, y).
    :param obj2_pixel_coords: Pixel coordinates of the second object (x, y).
    :param obj1_distance: Known distance of the first object from the camera.
    :param obj2_distance: Known distance of the second object from the camera.
    :return: Distance in meters between the two objects.
    """
    

    # Load camera calibration data
    with open("calibration_data.pkl", "rb") as f:
        camera_matrix, dist_coeffs = pickle.load(f)

    # Undistort the points
    pixel_points = np.array([obj1_pixel_coords, obj2_pixel_coords], dtype=np.float32)
    undistorted_points = cv.undistortPoints(pixel_points, camera_matrix, dist_coeffs, P=camera_matrix)

    # Extract undistorted points
    undistorted_points1 = undistorted_points[0][0]
    undistorted_points2 = undistorted_points[0][1]

    # Calculate pixel distance
    pixel_distance = np.linalg.norm(undistorted_points1 - undistorted_points2)

    # Calculate real-world distance using similar triangles
    real_world_distance = (obj1_distance * obj2_distance) / pixel_distance

    return real_world_distance

def calculateDistanceBetweenObjects(angle, obj1_distance, obj2_distance):
    """
    Calculate the distance between two objects based on their distances from the camera and the angle between them.

    :param angle: Angle between the two objects in degrees.
    :param obj1_distance: Distance of the first object from the camera.
    :param obj2_distance: Distance of the second object from the camera.
    :return: Distance between the two objects in meters.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle)

    # Calculate the distance between the two objects using the law of cosines
    distance = np.sqrt(obj1_distance**2 + obj2_distance**2 - 2 * obj1_distance * obj2_distance * np.cos(angle_radians))
    return distance

def calculateCenter(obj_pixel_coords):
    """
    Calculate the center coordinates of an object based on its pixel coordinates.
    
    :param obj_pixel_coords: Pixel coordinates of the object (x1, y1, x2, y2).
    :return: Center coordinates (center_x, center_y).
    """
    print(f"Calculating center for object with pixel coordinates: {obj_pixel_coords}")
    x1, y1, x2, y2 = obj_pixel_coords
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def calculateAngle(obj1_pixel_coords, obj2_pixel_coords):
    """
    Calculate the angle between two objects in an image based on their pixel coordinates.
    
    :param imageLoc: Path to the image file.
    :param obj1_pixel_coords: Pixel coordinates of the first object (x, y).
    :param obj2_pixel_coords: Pixel coordinates of the second object (x, y).
    :return: Angle in degrees between the two objects.
    """

    print(f"Calculating angle between objects with pixel coordinates: {obj1_pixel_coords} and {obj2_pixel_coords}")
    

    # Load camera calibration data
    with open("./calibration_data_master.pkl", "rb") as f:
        camera_matrixM, dist_coeffsM, rvecsM, tvecsM = pickle.load(f)

    print("Undistorting points...")
    # Undistort the image
    pixel_points = np.array([obj1_pixel_coords, obj2_pixel_coords], dtype=np.float32)
    undistorted_points = cv.undistortPoints(pixel_points, camera_matrixM, dist_coeffsM, P=camera_matrixM)

    print(f"Undistorted points: {undistorted_points}")
    # Extract undistorted points
    undistorted_points1 = undistorted_points[0][0]
    undistorted_points2 = undistorted_points[1][0]

    print("Normalising points...")
    # Normalise the points to the camera coordinate system
    normalised_points1 = (undistorted_points1[0] + camera_matrixM[0, 2]) / camera_matrixM[0, 0], (undistorted_points1[1] + camera_matrixM[1, 2]) / camera_matrixM[1, 1]
    normalised_points2 = (undistorted_points2[0] + camera_matrixM[0, 2]) / camera_matrixM[0, 0], (undistorted_points2[1] + camera_matrixM[1, 2]) / camera_matrixM[1, 1]

    print("Creating ray vectors...")
    # Create ray vectors from the normalized points
    ray_vector1 = np.array([normalised_points1[0], normalised_points1[1], 1.0])
    ray_vector2 = np.array([normalised_points2[0], normalised_points2[1], 1.0])

    print("Normalising ray vectors...")
    # Normalize the ray vectors
    unit_ray_vector1 = ray_vector1 / np.linalg.norm(ray_vector1)
    unit_ray_vector2 = ray_vector2 / np.linalg.norm(ray_vector2)

    print("Calculating angle...")
    # Calculate the angle between the two rays
    dot_product = np.dot(unit_ray_vector1, unit_ray_vector2)
    print(f"Dot product: {dot_product}")
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the value is within valid range for acos
    print(f"Clipped dot product: {dot_product}")
    angle_radians = np.arccos(dot_product) # Calculate the angle in radians
    print(f"Angle in radians: {angle_radians}")
    angle_degrees = np.degrees(angle_radians) # Convert radians to degrees

    return angle_degrees