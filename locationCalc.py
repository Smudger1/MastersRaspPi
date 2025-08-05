import cv2 as cv
import numpy as np
import pickle

def calculateDistance(imageLoc, obj1_pixel_coords, obj2_pixel_coords, obj1_distance, obj2_distance):
    """
    Calculate the distance between two objects in an image based on their pixel coordinates and known distances.
    
    :param imageLoc: Path to the image file.
    :param obj1_pixel_coords: Pixel coordinates of the first object (x, y).
    :param obj2_pixel_coords: Pixel coordinates of the second object (x, y).
    :param obj1_distance: Known distance of the first object from the camera.
    :param obj2_distance: Known distance of the second object from the camera.
    :return: Distance in meters between the two objects.
    """
    
    # Load the image
    image = cv.imread(imageLoc)

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


def calculateAngle(imageLoc, obj1_pixel_coords, obj2_pixel_coords):
    """
    Calculate the angle between two objects in an image based on their pixel coordinates.
    
    :param imageLoc: Path to the image file.
    :param obj1_pixel_coords: Pixel coordinates of the first object (x, y).
    :param obj2_pixel_coords: Pixel coordinates of the second object (x, y).
    :return: Angle in degrees between the two objects.
    """
    
    # Load the image
    image = cv.imread(imageLoc)

    obj1_pixel_coords = (0, 0)  # Replace with actual pixel coordinates of object 1
    obj2_pixel_coords = (100, 100)  # Replace with actual pixel coordinates of object 2

    # Load camera calibration data
    with open("calibration_data.pkl", "rb") as f:
        camera_matrix, dist_coeffs = pickle.load(f)

    # Undistort the image
    pixel_points = np.array([obj1_pixel_coords, obj2_pixel_coords], dtype=np.float32)
    undistorted_points = cv.undistortPoints(pixel_points, camera_matrix, dist_coeffs, P=camera_matrix)

    # Extract undistorted points
    undistorted_points1 = undistorted_points[0][0]
    undistorted_points2 = undistorted_points[0][1]

    # Normalize the points to the camera coordinate system
    normalised_points1 = (undistorted_points1[0] + camera_matrix[0, 2]) / camera_matrix[0, 0], (undistorted_points1[1] + camera_matrix[1, 2]) / camera_matrix[1, 1]
    normalised_points2 = (undistorted_points2[0] + camera_matrix[0, 2]) / camera_matrix[0, 0], (undistorted_points2[1] + camera_matrix[1, 2]) / camera_matrix[1, 1]

    # Create ray vectors from the normalized points
    ray_vector1 = np.array([normalised_points1[0], normalised_points1[1], 1.0])
    ray_vector2 = np.array([normalised_points2[0], normalised_points2[1], 1.0])

    # Normalize the ray vectors
    unit_ray_vector1 = ray_vector1 / np.linalg.norm(ray_vector1)
    unit_ray_vector2 = ray_vector2 / np.linalg.norm(ray_vector2)

    # Calculate the angle between the two rays
    dot_product = np.dot(unit_ray_vector1, unit_ray_vector2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the value is within valid range for acos
    angle_radians = np.arccos(dot_product) # Calculate the angle in radians
    angle_degrees = np.degrees(angle_radians) # Convert radians to degrees

    return angle_degrees