import numpy as np
import cv2 as cv
import glob
import pickle

print("###### Starting Setup for camera calibration ######")

chessboard_size = (9, 6)  # Number of inner corners per chessboard row and column
master_cam_image_size = (4608, 2592)  # Size of the images used for calibration
slave_cam_image_size = (640, 480)  # Size of the images used for calibration

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 21
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsM = [] # 2d points in image plane for Master camera
imgpointsS = [] # 2d points in image plane for Slave camera

print("###### Setup Complete. Starting calibration process ######")

# Load images for Master camera calibration
images_master = glob.glob('./calibImgs/calibImageMaster_*.jpg')
images_slave = glob.glob('./calibImgs/calibImageSlave_*.jpg')

if not images_master or not images_slave:
    print("No images found for Master or Slave camera calibration.")
    exit()

for fname in images_master:
    print(f"Processing {fname}...")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Finding chessboard corners...")
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        print(f"Chessboard corners not found in {fname}. Skipping this image.")
        continue
    if ret == True:
        objpoints.append(objp)
        imgpointsM.append(corners)
        cv.drawChessboardCorners(img, chessboard_size, corners, ret)
        # Optionally save the image with drawn corners for later inspection
        cv.imwrite(f'./calibImgs/processed_{fname.split("/")[-1]}', img)

for fname in images_slave:
    print(f"Processing {fname}...")
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("Finding chessboard corners...")
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if not ret:
        print(f"Chessboard corners not found in {fname}. Skipping this image.")
        continue
    if ret == True:
        objpoints.append(objp)
        imgpointsS.append(corners)
        cv.drawChessboardCorners(img, chessboard_size, corners, ret)
        # Optionally save the image with drawn corners for later inspection
        cv.imwrite(f'./calibImgs/processed_{fname.split("/")[-1]}', img)

print("Chessboard corners found in all images. Proceeding with calibration...")


retM, camera_matrixM, dist_coeffsM, rvecsM, tvecsM = cv.calibrateCamera(objpoints, imgpointsM, master_cam_image_size, None, None)
retS, camera_matrixS, dist_coeffsS, rvecsS, tvecsS = cv.calibrateCamera(objpoints, imgpointsS, slave_cam_image_size, None, None)

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC


criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, camera_matrix_stereo, dist_coeffs_stereo, rvecs_stereo, tvecs_stereo = cv.stereoCalibrate(
    objpoints, 
    imgpointsM, imgpointsS,
    camera_matrixM, dist_coeffsM,
    camera_matrixS, dist_coeffsS,
    master_cam_image_size, criteria_stereo, flags
)

if retStereo:
    print("Calibration successful.")
    print("Camera matrix:")
    print(camera_matrix_stereo)
    print("Distortion coefficients:")
    print(dist_coeffs_stereo)

    # Save the calibration data to a file
    with open('./calibImgs/calibration_data.pkl', 'wb') as f:
        pickle.dump((camera_matrix_stereo, dist_coeffs_stereo), f)


print("Calibration data saved to './calibImgs/calibration_data.pkl'.")
print("###### Calibration process completed successfully! ######")


