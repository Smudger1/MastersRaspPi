import numpy as np
import cv2 as cv
import glob
import pickle

chessboard_size = (10, 7)  # Number of inner corners per chessboard row and column
image_size = (4608, 2592)  # Size of the images used for calibration

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 2
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./calibImgs/calibImage*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv.drawChessboardCorners(img, chessboard_size, corners, ret)
    cv.imshow('img', img)
    cv.waitKey(500)

cv.destroyAllWindows()
