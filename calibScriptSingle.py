import numpy as np
import cv2 as cv
import glob
import pickle

print("###### Starting Setup for camera calibration ######")

chessboard_size = (12, 8)  # Number of inner corners per chessboard row and column

image_size = (1280, 720)  # Set the image size for both cameras

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 21
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsM = [] # 2d points in image plane for Master camera

print("###### Setup Complete. Starting calibration process ######")

# Load images for Master camera calibration
images_master = glob.glob('./calibImgs/calibImageMaster_*.jpg')

if not images_master:
    print("No images found for Master or Slave camera calibration.")
    exit()

images_master.sort()

# Pair images by index (assuming same number and order)
num_pairs = len(images_master)
for i in range(num_pairs):
    fname_master = images_master[i]

    img_master = cv.imread(fname_master)

    gray_master = cv.cvtColor(img_master, cv.COLOR_BGR2GRAY)

    retM, cornersM = cv.findChessboardCorners(gray_master, chessboard_size, None)

    if retM:
        objpoints.append(objp)
        imgpointsM.append(cornersM)
        cv.drawChessboardCorners(img_master, chessboard_size, cornersM, retM)
        cv.imwrite(f'./calibImgs/processed_{fname_master.split("/")[-1]}', img_master)
    else:
        print(f"Skipping pair {fname_master} (corners not found in image)")

print("Chessboard corners found in all images. Proceeding with calibration...")


retM, camera_matrixM, dist_coeffsM, rvecsM, tvecsM = cv.calibrateCamera(objpoints, imgpointsM, image_size, None, None)

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC

with open('./calibration_data_master.pkl', 'wb') as f:
    pickle.dump((camera_matrixM, dist_coeffsM, rvecsM, tvecsM), f)


criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)



print("Calibration data saved to './calibration_data_master.pkl'.")
print("###### Calibration process completed successfully! ######")


