import numpy as np
import cv2 as cv
import glob
import pickle

print("###### Starting Setup for camera calibration ######")

chessboard_size = (12, 8)  # Number of inner corners per chessboard row and column
master_cam_image_size = (4608, 2592)  # Size of the images used for calibration
slave_cam_image_size = (640, 480)  # Size of the images used for calibration

image_size = (1280, 720)  # Set the image size for both cameras

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

images_master.sort()
images_slave.sort()

# Pair images by index (assuming same number and order)
num_pairs = min(len(images_master), len(images_slave))
for i in range(num_pairs):
    fname_master = images_master[i]
    fname_slave = images_slave[i]

    img_master = cv.imread(fname_master)
    img_slave = cv.imread(fname_slave)

    gray_master = cv.cvtColor(img_master, cv.COLOR_BGR2GRAY)
    gray_slave = cv.cvtColor(img_slave, cv.COLOR_BGR2GRAY)

    retM, cornersM = cv.findChessboardCorners(gray_master, chessboard_size, None)
    retS, cornersS = cv.findChessboardCorners(gray_slave, chessboard_size, None)

    if retM and retS:
        objpoints.append(objp)
        imgpointsM.append(cornersM)
        imgpointsS.append(cornersS)
        cv.drawChessboardCorners(img_master, chessboard_size, cornersM, retM)
        cv.drawChessboardCorners(img_slave, chessboard_size, cornersS, retS)
        cv.imwrite(f'./calibImgs/processed_{fname_master.split("/")[-1]}', img_master)
        cv.imwrite(f'./calibImgs/processed_{fname_slave.split("/")[-1]}', img_slave)
    else:
        print(f"Skipping pair {fname_master}, {fname_slave} (corners not found in both)")

print("Chessboard corners found in all images. Proceeding with calibration...")


retM, camera_matrixM, dist_coeffsM, rvecsM, tvecsM = cv.calibrateCamera(objpoints, imgpointsM, image_size, None, None)
retS, camera_matrixS, dist_coeffsS, rvecsS, tvecsS = cv.calibrateCamera(objpoints, imgpointsS, image_size, None, None)

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC


criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, camera_matrix_stereo, dist_coeffs_stereo, rvecs_stereo, tvecs_stereo = cv.stereoCalibrate(
    objpoints, 
    imgpointsM, imgpointsS,
    camera_matrixM, dist_coeffsM,
    camera_matrixS, dist_coeffsS,
    image_size, criteria_stereo, flags
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


